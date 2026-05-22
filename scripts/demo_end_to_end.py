#!/usr/bin/env python3
"""
End-to-end cocoa resilience demo v5 (Round-5 modules, single JSON + markdown).

Chains EUDR, TerraMind+TiM exposure, ATTRICI attribution, intervention MC/CQR with
mediation, DVDS sensitivity, WCTM drift on scenarios, optional CorrDiff, and DR policy rules.

Example::

    python scripts/demo_end_to_end.py --mock-gee --pretty
    USE_REAL_FEATURES=false python scripts/demo_end_to_end.py --out reports/demo/e2e_civ_v5.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from analysis.policy_targeting import learn_policy_tree, render_policy_rules
from api.config import APISettings
from api.cqr_loader import load_cqr_bundle
from api.eudr import EudrDueDiligenceRequest, EudrStatusBlock, run_eudr_due_diligence
from api.feature_resolver import build_resolver_from_settings
from api.financial import calculate_financial_impact
from api.model_loader import load_casej_model, load_yield_model
from api.online_conformal_store import build_store_from_settings
from api.schemas import (
    FarmLocation,
    InterventionType,
    SimulateClimateAttributionRequest,
    SimulateInterventionRequest,
    SimulateScenarioRequest,
)
from api.simulation import (
    simulate_climate_attribution,
    simulate_intervention,
    simulate_scenario,
)
from counterfactual.corrdiff_downscaler import corrdiff_cache_path, write_synthetic_corrdiff_cache
from data.alphaearth_embeddings import AEF_ANNUAL_COLLECTION, AEF_ATTRIBUTION
from data.cocoa_exposure import (
    FDP_COCOA_COLLECTION,
    FDP_MODEL_CARD_URL,
    sample_cocoa_probability_at_point,
)
from data.era5_ingest import CHIRPS_DAILY, ERA5_DAILY, ERA5_LAND_DAILY
from data.farm_panel import load_synthetic_panel
from data.whisp_client import WHISP_PORTAL_URL, MockWhispClient, WhispClient
from monitoring.drift_store import build_drift_store_from_settings

logger = logging.getLogger(__name__)

SAMPLE_CIV_POLYGON: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [
            [-5.345678, 6.123456],
            [-5.344678, 6.123456],
            [-5.344678, 6.122456],
            [-5.345678, 6.122456],
            [-5.345678, 6.123456],
        ]
    ],
}

SAMPLE_LAT = 6.123
SAMPLE_LON = -5.345
DEFAULT_FARM_SIZE_HA = 3.2
DEFAULT_CURRENT_YIELD_T_HA = 1.8
DEFAULT_INTERVENTION = InterventionType.shade_trees
DEFAULT_CLIMATE_YEAR = 2023
DEFAULT_SCENARIO = "ssp585"
DEFAULT_HORIZON_YEAR = 2050
DEFAULT_V5_JSON = _REPO_ROOT / "reports" / "demo" / "e2e_civ_v5.json"
DEFAULT_V5_MD = _REPO_ROOT / "reports" / "demo" / "e2e_civ_v5.md"
CORRDIFF_REGION = "civ"


def polygon_centroid(polygon: dict[str, Any]) -> tuple[float, float]:
    from shapely.geometry import shape

    geom = shape(polygon)
    return float(geom.centroid.y), float(geom.centroid.x)


def build_demo_settings(*, mock_gee: bool) -> APISettings:
    settings = APISettings()
    if mock_gee:
        os.environ["USE_REAL_FEATURES"] = "false"
        settings = settings.model_copy(
            update={"use_real_features": False, "mediation_n_bootstrap": 80}
        )
    return settings


def write_era5_demo_zarr(
    path: Path, lat: float, lon: float, year: int = DEFAULT_CLIMATE_YEAR
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(f"{year}-01-01", periods=365, freq="D")
    seasonal = np.sin(2 * np.pi * np.arange(365) / 365.0)
    tmax = (31.0 + 2.0 * seasonal).astype(np.float32)
    tmin = (tmax - 7.0).astype(np.float32)
    tmean = (0.5 * (tmax + tmin)).astype(np.float32)
    precip = np.clip(np.abs(np.random.default_rng(7).normal(4.0, 2.0, 365)), 0, 60).astype(
        np.float32
    )
    srad = (18.0 + 2.0 * seasonal).astype(np.float32)
    rh = np.clip(78.0 + 5.0 * seasonal, 5, 100).astype(np.float32)
    wind = np.full(365, 2.0, dtype=np.float32)
    sm = np.full(365, 0.28, dtype=np.float32)
    vpd = np.full(365, 1.1, dtype=np.float32)
    et0 = np.full(365, 3.8, dtype=np.float32)
    cwd = (et0 - precip).astype(np.float32)
    shape = (365, 1, 1)
    ds = xr.Dataset(
        {
            "tmax": (("time", "latitude", "longitude"), tmax.reshape(shape)),
            "tmin": (("time", "latitude", "longitude"), tmin.reshape(shape)),
            "tmean": (("time", "latitude", "longitude"), tmean.reshape(shape)),
            "precip": (("time", "latitude", "longitude"), precip.reshape(shape)),
            "srad": (("time", "latitude", "longitude"), srad.reshape(shape)),
            "rh_mean": (("time", "latitude", "longitude"), rh.reshape(shape)),
            "wind10m": (("time", "latitude", "longitude"), wind.reshape(shape)),
            "sm_root": (("time", "latitude", "longitude"), sm.reshape(shape)),
            "vpd_mean": (("time", "latitude", "longitude"), vpd.reshape(shape)),
            "et0": (("time", "latitude", "longitude"), et0.reshape(shape)),
            "cwd": (("time", "latitude", "longitude"), cwd.reshape(shape)),
            "co2_ppm": (("time", "latitude", "longitude"), np.full(shape, 420.0, dtype=np.float32)),
        },
        coords={
            "time": times,
            "latitude": np.array([lat], dtype=np.float32),
            "longitude": np.array([lon], dtype=np.float32),
        },
    )
    ds["cwd_cum"] = ds["cwd"].cumsum(dim="time")
    ds.to_zarr(path, mode="w")


def write_cmip6_demo_zarr(path: Path, hist: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fut = hist.isel(time=slice(0, 60)).copy()
    fut = fut.assign_coords(time=pd.date_range("2050-01-01", periods=60, freq="D"))
    raw = xr.Dataset(
        {
            "tas": fut["tmean"] + 273.15 + 2.5,
            "tasmax": fut["tmax"] + 273.15 + 2.5,
            "tasmin": fut["tmin"] + 273.15 + 2.5,
            "hurs": fut["rh_mean"] - 3.0,
            "pr": (fut["precip"] * 0.85) / 86400.0,
            "rsds": (fut["srad"] * 1.08) * 1e6 / 86400.0,
            "sfcWind": fut["wind10m"] * 1.05,
        }
    )
    raw = raw.expand_dims(model=["ensemble"], scenario=["ssp245", "ssp585"])
    raw.to_zarr(path, mode="w")


def write_attrici_counterfactual_zarr(
    factual_path: Path,
    output_path: Path,
    *,
    year: int = DEFAULT_CLIMATE_YEAR,
) -> None:
    factual = xr.open_zarr(factual_path, consolidated=False)
    annual = factual.sel(time=factual["time"].dt.year == year).isel(time=slice(0, 365))
    cf = annual.copy(deep=False)
    for var in ("tmax", "tmin", "precip", "rh_mean", "srad", "wind10m"):
        if var not in annual:
            continue
        delta = 1.2 if var in ("tmax", "tmin") else 0.0
        cf[f"{var}_cf"] = (annual[var] - delta).astype(np.float32)
    out = cf[[v for v in cf.data_vars if v.endswith("_cf")]]
    out.attrs["method"] = "demo_synthetic_attrici_counterfactual"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_zarr(output_path, mode="w")


def ensure_demo_datastores(settings: APISettings, lat: float, lon: float) -> None:
    era5 = Path(settings.era5_zarr_path)
    if not era5.is_dir():
        logger.info("Writing demo ERA5 Zarr → %s", era5)
        write_era5_demo_zarr(era5, lat, lon)
    cmip = Path(settings.cmip6_zarr_path)
    if not cmip.is_dir():
        logger.info("Writing demo CMIP6 Zarr → %s", cmip)
        hist = xr.open_zarr(era5, consolidated=False)
        write_cmip6_demo_zarr(cmip, hist)
    cf = Path(settings.era5_counterfactual_zarr_path)
    if not cf.is_dir():
        logger.info("Writing demo ATTRICI counterfactual Zarr → %s", cf)
        write_attrici_counterfactual_zarr(era5, cf)


def build_source_attributions(
    *,
    exposure_backend: str,
    uq_method: str,
) -> list[dict[str, str]]:
    base = [
        {
            "id": "whisp",
            "role": "EUDR deforestation screening",
            "citation": f"Open Foris Whisp ({WHISP_PORTAL_URL})",
        },
        {
            "id": "eudr_regulation",
            "role": "Legal baseline (forest cutoff 2020-12-31)",
            "citation": "EU Regulation (EU) 2023/1115",
        },
        {
            "id": "fdp_cocoa_2025a",
            "role": "Cocoa exposure probability",
            "citation": f"FDP 2025a ({FDP_MODEL_CARD_URL})",
            "asset": FDP_COCOA_COLLECTION,
        },
        {
            "id": "aef",
            "role": "Cocoa exposure (embeddings)",
            "citation": AEF_ATTRIBUTION,
            "asset": AEF_ANNUAL_COLLECTION,
        },
        {
            "id": "terramind_tim",
            "role": "TerraMind 1.0 + TiM exposure sample",
            "citation": "IBM-ESA TerraMind-1.0-base + TiM path",
        },
        {
            "id": "era5_land",
            "role": "Historical daily climate",
            "citation": "ECMWF ERA5-Land",
            "asset": ERA5_LAND_DAILY,
        },
        {"id": "era5", "role": "Daily Tmax/Tmin", "asset": ERA5_DAILY},
        {"id": "chirps", "role": "Daily precipitation", "asset": CHIRPS_DAILY},
        {
            "id": "cmip6",
            "role": f"Future climate ({DEFAULT_SCENARIO} {DEFAULT_HORIZON_YEAR})",
            "citation": "CMIP6 delta-change",
        },
        {
            "id": "attrici",
            "role": "No-climate-change counterfactual",
            "citation": "Mengel et al. (2021) ATTRICI",
        },
        {
            "id": "yield_surrogate",
            "role": "Attribution + intervention + mediation",
            "citation": "YieldSurrogate v2",
        },
        {
            "id": "casej_surrogate",
            "role": "SSP scenario yields",
            "citation": "CASEJ neural surrogate",
        },
        {
            "id": "wctm_drift",
            "role": "Conformal drift monitoring",
            "citation": "WCTM (WATCH, ICML 2025)",
        },
        {
            "id": "dvds",
            "role": "Cooperative ATE sensitivity bounds",
            "citation": "Tan MSM / DVDS (Dorn et al. 2022)",
        },
        {
            "id": "corrdiff",
            "role": "Optional CorrDiff-CMIP6 downscaling",
            "citation": "NVIDIA CorrDiff + Earth2Studio",
        },
        {
            "id": "policy_tree",
            "role": "Honest DR targeting rules",
            "citation": "DRPolicyTree (econml)",
        },
        {
            "id": "mediation",
            "role": "NDE/NIE path decomposition",
            "citation": "Imai-Keele-Yamamoto g-computation",
        },
        {
            "id": "icco_pricing",
            "role": "Financial valuation",
            "citation": "ICCO / farm-gate pass-through",
        },
        {"id": "uq_method", "role": "90% uncertainty interval", "citation": uq_method.upper()},
        {
            "id": "exposure_backend",
            "role": "Default exposure resolver",
            "citation": exposure_backend,
        },
    ]
    return base


def _run_corrdiff_demo_section(
    settings: APISettings,
    farm_loc: FarmLocation,
    *,
    farm_size_ha: float,
    current_yield_t_ha: float,
    intervention_type: InterventionType,
    cocoa_price_usd: float,
    polygon: dict[str, Any],
    climate_year: int,
    casej_model: Any,
    feature_resolver: Any,
) -> dict[str, Any]:
    enabled = os.environ.get("CORRDIFF_AVAILABLE", "").lower() in ("1", "true", "yes")
    if not enabled:
        return {"status": "skipped", "reason": "CORRDIFF_AVAILABLE not set"}

    cache = corrdiff_cache_path(
        settings.corrdiff_processed_dir,
        DEFAULT_SCENARIO,
        DEFAULT_HORIZON_YEAR,
        CORRDIFF_REGION,
    )
    if not cache.is_dir():
        try:
            write_synthetic_corrdiff_cache(
                cache,
                scenario=DEFAULT_SCENARIO,
                horizon=DEFAULT_HORIZON_YEAR,
                region=CORRDIFF_REGION,
                n_samples=2,
                n_days=60,
            )
        except Exception as exc:
            return {"status": "skipped", "reason": f"cache write failed: {exc}"}

    req = SimulateScenarioRequest(
        farm_location=farm_loc,
        farm_size_ha=farm_size_ha,
        current_yield=current_yield_t_ha,
        intervention_type=intervention_type,
        cocoa_price_usd=cocoa_price_usd,
        country_code="CIV",
        farm_polygon=polygon,
        scenario=DEFAULT_SCENARIO,
        horizon_year=DEFAULT_HORIZON_YEAR,
        downscaling_method="corrdiff",
    )
    try:
        resp = simulate_scenario(
            req,
            casej_model,
            feature_resolver,
            historical_zarr_path=Path(settings.era5_zarr_path),
            cmip6_zarr_path=Path(settings.cmip6_zarr_path),
            climate_year=climate_year,
            settings=settings,
        )
        return {
            "status": "ok",
            "downscaling_method": resp.downscaling_method,
            "corrdiff_samples_used": resp.corrdiff_samples_used,
            "avoided_loss_tonnes_mean": resp.avoided_loss_tonnes.mean,
        }
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}


def _run_policy_targeting_demo() -> dict[str, Any]:
    panel = load_synthetic_panel(n_farms=400, n_years=6, treatment_year=3, seed=11)
    covariates = ["soil_quality_index", "historical_rainfall", "farm_size_ha"]
    result = learn_policy_tree(
        panel,
        treatment_col="received_intervention",
        outcome_col="yield_tonnes_per_ha",
        covariate_cols=covariates,
        n_bootstrap=0,
        min_samples_leaf=80,
        max_depth=3,
    )
    rules = render_policy_rules(result)[:3]
    return {
        "policy_value": result.policy_value,
        "n_rules_shown": len(rules),
        "rules": rules,
        "covariates": covariates,
    }


def write_markdown_summary(payload: dict[str, Any], path: Path) -> None:
    farm = payload.get("farm", {})
    lines = [
        "# Resilient Cocoa demo — Côte d'Ivoire (v5)",
        "",
        f"**Location:** {farm.get('lat')}, {farm.get('lon')} · **Area:** {farm.get('farm_size_ha')} ha",
        "",
        "## Executive summary",
        "",
        f"- **EUDR risk:** {payload.get('eudr_status', {}).get('risk_class', 'n/a')}",
        f"- **Climate-attributed loss:** {payload.get('climate_attributed_loss_t_per_ha', 0):.3f} t/ha",
        f"- **Shade-tree avoided loss:** {payload.get('intervention_avoided_loss_t_per_ha', 0):.3f} t/ha",
        f"- **Total avoided value (90%):** ${payload.get('total_avoided_loss_usd', {}).get('point', 0):,.0f}",
        f"- **Cocoa exposure (default backend):** {payload.get('cocoa_exposure_probability', 0):.2f}",
        "",
        "## Round-5 modules",
        "",
    ]
    tm = payload.get("exposure_terramind_tim", {})
    lines.append(f"1. **TerraMind+TiM exposure:** {tm.get('cocoa_probability', 'n/a')}")
    attr = payload.get("climate_attribution_detail", {})
    lines.append(
        f"2. **ATTRICI attribution:** factual {attr.get('factual_yield_t_per_ha', 0):.2f} vs CF {attr.get('counterfactual_yield_t_per_ha', 0):.2f} t/ha"
    )
    lines.append(
        f"3. **Intervention:** avoided {payload.get('intervention_avoided_loss_t_per_ha', 0):.3f} t/ha"
    )
    med = payload.get("mediation_shade_trees", {})
    if med:
        lines.append("4. **Mediation (shade trees):**")
        for m in med.get("per_mediator", []):
            lines.append(
                f"   - {m.get('mediator')}: NDE={m.get('nde', 0):.3f}, NIE={m.get('nie', 0):.3f}, "
                f"ρ*={m.get('rho_critical')}"
            )
    scen = payload.get("scenario_ssp585_2050", {})
    drift = scen.get("drift_status")
    lines.append(
        f"5. **SSP5-8.5 2050:** avoided loss mean {scen.get('avoided_loss_tonnes', {}).get('mean', 0):.2f} t"
    )
    if drift:
        lines.append(f"6. **WCTM drift:** alarm={drift.get('drift_alarm', False)}")
    dvds = payload.get("dvds_sensitivity", {})
    if dvds:
        lines.append(
            f"7. **DVDS (Λ={dvds.get('lambda', 1.5)}):** bounds [{dvds.get('ate_lower')}, {dvds.get('ate_upper')}]"
        )
    pol = payload.get("policy_targeting", {})
    if pol.get("rules"):
        lines.append("8. **Policy rules (top):**")
        for r in pol["rules"]:
            lines.append(f"   - {r}")
    cd = payload.get("scenario_corrdiff", {})
    lines.append(f"9. **CorrDiff scenario:** {cd.get('status', 'n/a')}")
    lines.extend(["", "---", "Generated by `scripts/demo_end_to_end.py` (v0.3.0)."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_end_to_end_demo(
    *,
    settings: APISettings | None = None,
    farm_polygon: dict[str, Any] | None = None,
    farm_size_ha: float = DEFAULT_FARM_SIZE_HA,
    current_yield_t_ha: float = DEFAULT_CURRENT_YIELD_T_HA,
    intervention_type: InterventionType = DEFAULT_INTERVENTION,
    cocoa_price_usd: float = 3200.0,
    whisp_client: WhispClient | None = None,
    mock_gee: bool = False,
    climate_year: int = DEFAULT_CLIMATE_YEAR,
) -> dict[str, Any]:
    settings = build_demo_settings(mock_gee=mock_gee) if settings is None else settings
    polygon = farm_polygon or SAMPLE_CIV_POLYGON
    lat, lon = polygon_centroid(polygon)

    if mock_gee or not Path(settings.era5_zarr_path).is_dir():
        ensure_demo_datastores(settings, lat, lon)

    whisp = whisp_client or MockWhispClient()
    eudr = await run_eudr_due_diligence(
        EudrDueDiligenceRequest(
            farm_polygon=polygon,
            commodity="cocoa",
            country_iso3="CIV",
            use_gee_fdp_screening=not mock_gee,
        ),
        settings=settings,
        whisp_client=whisp,
    )
    eudr_status = EudrStatusBlock(
        deforestation_post_2020=eudr.deforestation_post_2020,
        protected_area_overlap=eudr.protected_area_overlap,
        risk_class=eudr.risk_class,
        evidence_urls=eudr.evidence_urls,
        whisp_report_id=eudr.whisp_report_id,
        traceability=eudr.traceability,
    )

    exposure_default = float(
        sample_cocoa_probability_at_point(
            lat,
            lon,
            year=settings.cocoa_exposure_year,
            backend=settings.cocoa_exposure_backend,
            project=settings.earthengine_project,
        )
    )
    tim_settings = settings.model_copy(update={"cocoa_exposure_backend": "terramind_tim"})
    exposure_tim = float(
        sample_cocoa_probability_at_point(
            lat,
            lon,
            year=tim_settings.cocoa_exposure_year,
            backend="terramind_tim",
            project=tim_settings.earthengine_project,
        )
    )

    feature_resolver = build_resolver_from_settings(settings)
    yield_model = load_yield_model(settings.model_checkpoint_path, settings=settings)
    casej_model = load_casej_model(settings.casej_checkpoint_path, settings=settings)
    cqr_model, cqr_calibrator = load_cqr_bundle(settings)
    uq_method = settings.resolved_uq_method()
    use_cqr = uq_method == "cqr" and cqr_model is not None and cqr_calibrator is not None

    scenario_store = build_store_from_settings(settings)
    drift_store = build_drift_store_from_settings(settings)

    farm_loc = FarmLocation(lat=lat, lon=lon)
    climate_resp = simulate_climate_attribution(
        SimulateClimateAttributionRequest(
            farm_location=farm_loc,
            farm_size_ha=farm_size_ha,
            current_yield=current_yield_t_ha,
            intervention_type=intervention_type,
            cocoa_price_usd=cocoa_price_usd,
            country_code="CIV",
            climate_year=climate_year,
        ),
        yield_model,
        feature_resolver,
        counterfactual_zarr_path=Path(settings.era5_counterfactual_zarr_path),
        climate_year=climate_year,
    )

    intervention_resp = simulate_intervention(
        SimulateInterventionRequest(
            farm_location=farm_loc,
            farm_size_ha=farm_size_ha,
            current_yield=current_yield_t_ha,
            intervention_type=intervention_type,
            cocoa_price_usd=cocoa_price_usd,
            country_code="CIV",
            farm_polygon=polygon,
            include_sensitivity=True,
            decompose_mediators=["microclimate", "soil_moisture", "cssvd_prevalence"],
        ),
        yield_model,
        feature_resolver,
        uq_method=uq_method,
        cqr_model=cqr_model if use_cqr else None,
        cqr_calibrator=cqr_calibrator if use_cqr else None,
        settings=settings,
    )

    ci = intervention_resp.confidence_interval.avoided_loss_tonnes
    climate_tonnes = climate_resp.attributed_loss_tonnes_per_ha * farm_size_ha
    total_fin = calculate_financial_impact(
        climate_resp.total_avoided_loss_tonnes,
        cocoa_price_usd=cocoa_price_usd,
        country_code="CIV",
        lat=lat,
        lon=lon,
        ci_low_tonnes=climate_tonnes + ci.lower,
        ci_high_tonnes=climate_tonnes + ci.upper,
    )

    scenario_resp = simulate_scenario(
        SimulateScenarioRequest(
            farm_location=farm_loc,
            farm_size_ha=farm_size_ha,
            current_yield=current_yield_t_ha,
            intervention_type=intervention_type,
            cocoa_price_usd=cocoa_price_usd,
            country_code="CIV",
            farm_polygon=polygon,
            scenario=DEFAULT_SCENARIO,
            horizon_year=DEFAULT_HORIZON_YEAR,
        ),
        casej_model,
        feature_resolver,
        historical_zarr_path=Path(settings.era5_zarr_path),
        cmip6_zarr_path=Path(settings.cmip6_zarr_path),
        climate_year=climate_year,
        settings=settings,
        scenario_conformal_store=scenario_store,
        drift_store=drift_store,
    )

    dvds_slice: dict[str, Any] = {}
    if intervention_resp.sensitivity_bounds:
        for row in intervention_resp.sensitivity_bounds:
            if getattr(row, "lambda_", None) == 1.5 or getattr(row, "lambda", None) == 1.5:
                dvds_slice = row.model_dump()
                break
        if not dvds_slice and intervention_resp.sensitivity_bounds:
            dvds_slice = intervention_resp.sensitivity_bounds[0].model_dump()

    mediation_dump: dict[str, Any] | None = None
    if intervention_resp.mediation is not None:
        mediation_dump = {
            "per_mediator": [m.model_dump() for m in intervention_resp.mediation.per_mediator],
            "path_table": intervention_resp.mediation.path_table,
        }

    intervention_avoided_t_ha = climate_resp.intervention_avoided_loss_tonnes / farm_size_ha
    drift_status = None
    if scenario_resp.drift_status is not None:
        drift_status = scenario_resp.drift_status.model_dump()

    return {
        "version": "0.3.0",
        "farm": {
            "lat": lat,
            "lon": lon,
            "farm_size_ha": farm_size_ha,
            "country_code": "CIV",
            "climate_reference_year": climate_year,
        },
        "climate_attributed_loss_t_per_ha": climate_resp.attributed_loss_tonnes_per_ha,
        "intervention_avoided_loss_t_per_ha": intervention_avoided_t_ha,
        "total_avoided_loss_tonnes": climate_resp.total_avoided_loss_tonnes,
        "total_avoided_loss_usd": {
            "point": total_fin.usd.point,
            "ci_low": total_fin.usd.ci_low,
            "ci_high": total_fin.usd.ci_high,
            "level": 0.9,
            "method": uq_method,
        },
        "cocoa_exposure_probability": exposure_default,
        "exposure_terramind_tim": {"cocoa_probability": exposure_tim, "backend": "terramind_tim"},
        "eudr_status": eudr_status.model_dump(),
        "scenario_ssp585_2050": {
            "baseline_yield_t_per_ha": scenario_resp.baseline_yield_tonnes_per_ha.model_dump(),
            "projected_yield_t_per_ha": scenario_resp.projected_yield_tonnes_per_ha.model_dump(),
            "avoided_loss_tonnes": scenario_resp.avoided_loss_tonnes.model_dump(),
            "financial_impact_usd_mean": scenario_resp.financial_impact_usd_mean,
            "drift_status": drift_status,
            "drift_alarm": scenario_resp.drift_alarm,
        },
        "scenario_corrdiff": _run_corrdiff_demo_section(
            settings,
            farm_loc,
            farm_size_ha=farm_size_ha,
            current_yield_t_ha=current_yield_t_ha,
            intervention_type=intervention_type,
            cocoa_price_usd=cocoa_price_usd,
            polygon=polygon,
            climate_year=climate_year,
            casej_model=casej_model,
            feature_resolver=feature_resolver,
        ),
        "dvds_sensitivity": dvds_slice,
        "policy_targeting": _run_policy_targeting_demo(),
        "mediation_shade_trees": mediation_dump,
        "climate_attribution_detail": {
            "factual_yield_t_per_ha": climate_resp.factual_yield_tonnes_per_ha,
            "counterfactual_yield_t_per_ha": climate_resp.counterfactual_yield_tonnes_per_ha,
        },
        "source_attributions": build_source_attributions(
            exposure_backend=settings.cocoa_exposure_backend,
            uq_method=uq_method,
        ),
        "data_paths": {
            "era5_zarr": str(settings.era5_zarr_path),
            "era5_counterfactual_zarr": str(settings.era5_counterfactual_zarr_path),
            "cmip6_zarr": str(settings.cmip6_zarr_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end cocoa resilience demo (v5)")
    parser.add_argument("--out", type=Path, default=DEFAULT_V5_JSON, help="JSON summary path")
    parser.add_argument(
        "--legacy-out",
        type=Path,
        default=None,
        help="Also write legacy e2e_civ.json filename",
    )
    parser.add_argument(
        "--md-out", type=Path, default=None, help="Markdown summary (default: sibling .md)"
    )
    parser.add_argument("--mock-gee", action="store_true", help="Offline geo_mock + synthetic Zarr")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    payload = asyncio.run(run_end_to_end_demo(mock_gee=args.mock_gee))
    text = json.dumps(payload, indent=2 if args.pretty else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    md_path = args.md_out or args.out.with_suffix(".md")
    write_markdown_summary(payload, md_path)
    if args.legacy_out:
        args.legacy_out.parent.mkdir(parents=True, exist_ok=True)
        args.legacy_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    logger.info("Wrote demo summary → %s and %s", args.out, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
