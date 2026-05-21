#!/usr/bin/env python3
"""
End-to-end cocoa resilience demo (single JSON artifact).

Chains EUDR (Whisp), FDP/AEF exposure, ERA5 + CMIP6 SSP5-8.5 2050, ATTRICI counterfactual,
yield surrogate climate attribution, CASEJ scenario simulation, and financial valuation.

Example::

    python scripts/demo_end_to_end.py --out reports/demo/e2e_civ.json
    USE_REAL_FEATURES=false python scripts/demo_end_to_end.py --mock-gee --pretty
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

from api.config import APISettings
from api.cqr_loader import load_cqr_bundle
from api.eudr import EudrDueDiligenceRequest, EudrStatusBlock, run_eudr_due_diligence
from api.feature_resolver import build_resolver_from_settings
from api.financial import calculate_financial_impact
from api.model_loader import load_casej_model, load_yield_model
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
from data.alphaearth_embeddings import AEF_ANNUAL_COLLECTION, AEF_ATTRIBUTION
from data.cocoa_exposure import (
    FDP_COCOA_COLLECTION,
    FDP_MODEL_CARD_URL,
    sample_cocoa_probability_at_point,
)
from data.era5_ingest import CHIRPS_DAILY, ERA5_DAILY, ERA5_LAND_DAILY
from data.whisp_client import WHISP_DOCS_URL, WHISP_PORTAL_URL, MockWhispClient, WhispClient
from models.yield_surrogate import CLIMATE_CHANNEL_NAMES

logger = logging.getLogger(__name__)

# Sample farm near Divo, Côte d'Ivoire (Kalischek / FDP validation belt)
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


def polygon_centroid(polygon: dict[str, Any]) -> tuple[float, float]:
    from shapely.geometry import shape

    geom = shape(polygon)
    return float(geom.centroid.y), float(geom.centroid.x)


def write_era5_demo_zarr(path: Path, lat: float, lon: float, year: int = DEFAULT_CLIMATE_YEAR) -> None:
    """Write a minimal ERA5-Land daily Zarr at one grid cell (365 days)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(f"{year}-01-01", periods=365, freq="D")
    seasonal = np.sin(2 * np.pi * np.arange(365) / 365.0)
    tmax = (31.0 + 2.0 * seasonal).astype(np.float32)
    tmin = (tmax - 7.0).astype(np.float32)
    tmean = (0.5 * (tmax + tmin)).astype(np.float32)
    precip = np.clip(np.abs(np.random.default_rng(7).normal(4.0, 2.0, 365)), 0, 60).astype(np.float32)
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
    """Minimal CMIP6 ensemble cube for ScenarioBuilder (warmer SSP5-8.5 2050 window)."""
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
    lat: float,
    lon: float,
    year: int = DEFAULT_CLIMATE_YEAR,
    use_fast_attrici: bool = False,
) -> None:
    """
    Build counterfactual Zarr for climate attribution (``*_cf`` bands).

    Offline demo writes a cooler synthetic world; set ``use_fast_attrici=True`` with
    multi-year factual data to run :class:`~data.attrici_fast_detrend.FastATTRICICounterfactual`.
    """
    del lat, lon
    factual = xr.open_zarr(factual_path, consolidated=False)
    annual = factual.sel(time=factual["time"].dt.year == year).isel(time=slice(0, 365))

    if use_fast_attrici:
        from data.attrici_fast_detrend import (
            FastATTRICICounterfactual,
            recompute_derived_counterfactuals,
        )

        year_vals = sorted({int(y) for y in annual.time.dt.year.values})
        gmt = pd.Series(
            np.linspace(0.0, 1.5, max(1, len(year_vals))),
            index=year_vals if year_vals else [year],
        )
        cf_ds = FastATTRICICounterfactual(
            gmt,
            variables=("tmax", "tmin", "precip", "rh_mean", "srad", "wind10m"),
        ).fit_transform(annual)
        cf_ds = recompute_derived_counterfactuals(cf_ds)
        out_vars = {k: v for k, v in cf_ds.data_vars.items() if k.endswith("_cf")}
        out = xr.Dataset(out_vars, coords=cf_ds.coords)
        out.attrs["method"] = "FastATTRICICounterfactual (Mengel et al. 2021)"
    else:
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


def ensure_demo_datastores(
    settings: APISettings,
    lat: float,
    lon: float,
) -> None:
    """Create ERA5, CMIP6, and ATTRICI Zarr stubs when missing (offline demo)."""
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
        write_attrici_counterfactual_zarr(era5, cf, lat=lat, lon=lon)


def build_source_attributions(
    *,
    exposure_backend: str,
    uq_method: str,
) -> list[dict[str, str]]:
    """Citations for every dataset touched in the demo pipeline."""
    return [
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
            "citation": f"Forest Data Partnership model 2025a ({FDP_MODEL_CARD_URL})",
            "asset": FDP_COCOA_COLLECTION,
        },
        {
            "id": "aef",
            "role": "Cocoa exposure (embeddings + head)",
            "citation": AEF_ATTRIBUTION,
            "asset": AEF_ANNUAL_COLLECTION,
        },
        {
            "id": "era5_land",
            "role": "Historical daily climate",
            "citation": "ECMWF ERA5-Land via Google Earth Engine",
            "asset": ERA5_LAND_DAILY,
        },
        {
            "id": "era5",
            "role": "Daily Tmax/Tmin",
            "asset": ERA5_DAILY,
        },
        {
            "id": "chirps",
            "role": "Daily precipitation (when enabled in ingest)",
            "asset": CHIRPS_DAILY,
        },
        {
            "id": "cmip6",
            "role": f"Future climate deltas ({DEFAULT_SCENARIO} {DEFAULT_HORIZON_YEAR})",
            "citation": "CMIP6 delta-change on ERA5 (ScenarioBuilder)",
        },
        {
            "id": "attrici",
            "role": "No-climate-change counterfactual",
            "citation": "Mengel et al. (2021) ATTRICI fast detrend",
        },
        {
            "id": "yield_surrogate",
            "role": "Climate attribution + intervention MC/CQR",
            "citation": "Resilient Cocoa YieldSurrogateModel",
        },
        {
            "id": "casej_surrogate",
            "role": "SSP scenario factual vs counterfactual yields",
            "citation": "CASEJ neural surrogate (CO2-conditioned)",
        },
        {
            "id": "icco_pricing",
            "role": "Financial valuation",
            "citation": "ICCO / farm-gate pass-through (api.financial)",
        },
        {
            "id": "uq_method",
            "role": "90% uncertainty interval",
            "citation": uq_method.upper(),
        },
        {
            "id": "exposure_backend",
            "role": "Exposure resolver path",
            "citation": exposure_backend,
        },
    ]


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
    """
    Execute the full pipeline and return a JSON-serializable summary dict.
    """
    settings = settings or APISettings()
    polygon = farm_polygon or SAMPLE_CIV_POLYGON
    lat, lon = polygon_centroid(polygon)

    if mock_gee:
        os.environ["USE_REAL_FEATURES"] = "false"
        settings = settings.model_copy(update={"use_real_features": False})

    if mock_gee or not Path(settings.era5_zarr_path).is_dir():
        ensure_demo_datastores(settings, lat, lon)

    whisp = whisp_client or MockWhispClient()
    eudr_req = EudrDueDiligenceRequest(
        farm_polygon=polygon,
        commodity="cocoa",
        country_iso3="CIV",
        use_gee_fdp_screening=not mock_gee,
    )
    eudr = await run_eudr_due_diligence(eudr_req, settings=settings, whisp_client=whisp)
    eudr_status = EudrStatusBlock(
        deforestation_post_2020=eudr.deforestation_post_2020,
        protected_area_overlap=eudr.protected_area_overlap,
        risk_class=eudr.risk_class,
        evidence_urls=eudr.evidence_urls,
        whisp_report_id=eudr.whisp_report_id,
        traceability=eudr.traceability,
    )

    exposure_p = float(
        sample_cocoa_probability_at_point(
            lat,
            lon,
            year=settings.cocoa_exposure_year,
            backend=settings.cocoa_exposure_backend,
            project=settings.earthengine_project,
        )
    )

    feature_resolver = build_resolver_from_settings(settings)
    yield_model = load_yield_model(settings.model_checkpoint_path, settings=settings)
    casej_model = load_casej_model(settings.casej_checkpoint_path, settings=settings)
    cqr_model, cqr_calibrator = load_cqr_bundle(settings)
    uq_method = settings.resolved_uq_method()
    use_cqr = uq_method == "cqr" and cqr_model is not None and cqr_calibrator is not None

    farm_loc = FarmLocation(lat=lat, lon=lon)
    climate_req = SimulateClimateAttributionRequest(
        farm_location=farm_loc,
        farm_size_ha=farm_size_ha,
        current_yield=current_yield_t_ha,
        intervention_type=intervention_type,
        cocoa_price_usd=cocoa_price_usd,
        country_code="CIV",
        climate_year=climate_year,
    )
    climate_resp = simulate_climate_attribution(
        climate_req,
        yield_model,
        feature_resolver,
        counterfactual_zarr_path=Path(settings.era5_counterfactual_zarr_path),
        climate_year=climate_year,
    )

    intervention_req = SimulateInterventionRequest(
        farm_location=farm_loc,
        farm_size_ha=farm_size_ha,
        current_yield=current_yield_t_ha,
        intervention_type=intervention_type,
        cocoa_price_usd=cocoa_price_usd,
        country_code="CIV",
        farm_polygon=polygon,
    )
    intervention_resp = simulate_intervention(
        intervention_req,
        yield_model,
        feature_resolver,
        uq_method=uq_method,
        cqr_model=cqr_model if use_cqr else None,
        cqr_calibrator=cqr_calibrator if use_cqr else None,
        settings=None,
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

    scenario_req = SimulateScenarioRequest(
        farm_location=farm_loc,
        farm_size_ha=farm_size_ha,
        current_yield=current_yield_t_ha,
        intervention_type=intervention_type,
        cocoa_price_usd=cocoa_price_usd,
        country_code="CIV",
        farm_polygon=polygon,
        scenario=DEFAULT_SCENARIO,
        horizon_year=DEFAULT_HORIZON_YEAR,
    )
    scenario_resp = simulate_scenario(
        scenario_req,
        casej_model,
        feature_resolver,
        historical_zarr_path=Path(settings.era5_zarr_path),
        cmip6_zarr_path=Path(settings.cmip6_zarr_path),
        climate_year=climate_year,
        settings=None,
    )

    intervention_avoided_t_ha = climate_resp.intervention_avoided_loss_tonnes / farm_size_ha

    return {
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
        "cocoa_exposure_probability": exposure_p,
        "eudr_status": eudr_status.model_dump(),
        "scenario_ssp585_2050": {
            "baseline_yield_t_per_ha": scenario_resp.baseline_yield_tonnes_per_ha.model_dump(),
            "projected_yield_t_per_ha": scenario_resp.projected_yield_tonnes_per_ha.model_dump(),
            "avoided_loss_tonnes": scenario_resp.avoided_loss_tonnes.model_dump(),
            "financial_impact_usd_mean": scenario_resp.financial_impact_usd_mean,
        },
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
    parser = argparse.ArgumentParser(description="Run end-to-end cocoa resilience demo")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "reports" / "demo" / "e2e_civ.json",
        help="Write JSON summary to this path",
    )
    parser.add_argument(
        "--mock-gee",
        action="store_true",
        help="Use geo_mock + synthetic Zarr (no Earth Engine credentials)",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    payload = asyncio.run(run_end_to_end_demo(mock_gee=args.mock_gee))
    text = json.dumps(payload, indent=2 if args.pretty else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    logger.info("Wrote demo summary → %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
