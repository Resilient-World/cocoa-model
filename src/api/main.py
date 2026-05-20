"""FastAPI entrypoint for the Avoided Loss simulation service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
import pandas as pd

from api.config import APISettings
from api.cqr_loader import load_cqr_bundle
from api.feature_resolver import build_resolver_from_settings
from api.model_loader import load_casej_model, load_yield_model
from api.schemas import (
    ComplianceDdsRequest,
    ComplianceDdsResponse,
    RankInterventionsRequest,
    RankInterventionsResponse,
    SimulateClimateAttributionRequest,
    SimulateClimateAttributionResponse,
    SimulateInterventionRequest,
    SimulateInterventionResponse,
    SimulateScenarioRequest,
    SimulateScenarioResponse,
)
from api.simulation import (
    simulate_climate_attribution,
    simulate_intervention,
    simulate_scenario,
)
from analysis.heterogeneity import estimate_cate
from analysis.policy_targeting import rank_farms_by_uplift
from compliance.eudr import (
    DeforestationResult,
    assess_country_risk,
    check_deforestation_free,
    generate_dds,
    validate_geolocation,
)
from models.conformal import load_conformal_if_exists
from models.casej_surrogate import CASEJSurrogate
from models.yield_surrogate import YieldSurrogateModel


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = APISettings()
    app.state.settings = settings
    app.state.feature_resolver = build_resolver_from_settings(settings)
    app.state.yield_model = load_yield_model(
        settings.model_checkpoint_path,
        settings=settings,
    )
    app.state.casej_model = load_casej_model(
        settings.casej_checkpoint_path,
        settings=settings,
    )
    app.state.conformal = load_conformal_if_exists(settings.conformal_json_path)
    cqr_model, cqr_calibrator = load_cqr_bundle(settings)
    app.state.cqr_model = cqr_model
    app.state.cqr_calibrator = cqr_calibrator
    yield


app = FastAPI(
    title="Resilient Cocoa Model API",
    description="Geospatial ML inference and intervention simulation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/simulate-intervention",
    response_model=SimulateInterventionResponse,
    summary="Simulate intervention impact (avoided loss)",
)
def simulate_intervention_endpoint(
    request: SimulateInterventionRequest,
) -> SimulateInterventionResponse:
    """
    Resolve ERA5/static features, run yield surrogate inference for counterfactual
    and factual scenarios, and return avoided loss with a 90% confidence interval.
    """
    model: YieldSurrogateModel = app.state.yield_model
    settings: APISettings = app.state.settings

    try:
        return simulate_intervention(
            request,
            model,
            app.state.feature_resolver,
            num_samples=settings.mc_num_samples,
            yield_blend_weight=settings.yield_blend_weight,
            climate_year=settings.climate_reference_year,
            conformal=getattr(app.state, "conformal", None),
            uq_method=settings.resolved_uq_method(),
            cqr_model=getattr(app.state, "cqr_model", None),
            cqr_calibrator=getattr(app.state, "cqr_calibrator", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/simulate-climate-attribution",
    response_model=SimulateClimateAttributionResponse,
    summary="Decompose avoided loss into climate-attributed vs intervention components",
)
def simulate_climate_attribution_endpoint(
    request: SimulateClimateAttributionRequest,
) -> SimulateClimateAttributionResponse:
    """
    Compares factual ERA5 yields vs ATTRICI counterfactual (no anthropogenic forcing) and
    adds intervention avoided loss from the standard paired-forward path.
    """
    model: YieldSurrogateModel = app.state.yield_model
    settings: APISettings = app.state.settings
    try:
        return simulate_climate_attribution(
            request,
            model,
            app.state.feature_resolver,
            counterfactual_zarr_path=settings.era5_counterfactual_zarr_path,
            num_samples=settings.mc_num_samples,
            yield_blend_weight=settings.yield_blend_weight,
            climate_year=request.climate_year or settings.climate_reference_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/simulate-scenario",
    response_model=SimulateScenarioResponse,
    summary="Simulate intervention under CMIP6 SSP future climate (delta-change ERA5)",
)
def simulate_scenario_endpoint(request: SimulateScenarioRequest) -> SimulateScenarioResponse:
    """
    Applies NASA/GDDP-CMIP6 monthly deltas via ``ScenarioBuilder`` to the historical ERA5 Zarr,
    then runs paired Monte Carlo yields for baseline vs intervention with mean / p10 / p90 bands.
    """
    model: CASEJSurrogate = app.state.casej_model
    settings: APISettings = app.state.settings

    try:
        return simulate_scenario(
            request,
            model,
            app.state.feature_resolver,
            historical_zarr_path=settings.era5_zarr_path,
            cmip6_zarr_path=settings.cmip6_zarr_path,
            num_samples=settings.mc_num_samples,
            yield_blend_weight=settings.yield_blend_weight,
            climate_year=settings.climate_reference_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/rank-interventions",
    response_model=RankInterventionsResponse,
    summary="Rank farms by estimated uplift (CATE) for cooperative rollouts",
)
def rank_interventions_endpoint(request: RankInterventionsRequest) -> RankInterventionsResponse:
    """
    Estimate heterogeneous uplift using R-learner / tree ensemble methods on tabular data,
    then rank farms by net uplift in USD (cocoa price × avoided tonnes − cost).
    """
    df = pd.DataFrame(request.rows)
    if request.farm_area_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Missing farm area column '{request.farm_area_col}'")
    try:
        cate = estimate_cate(
            df,
            outcome=request.outcome,
            treatment=request.treatment,
            covariates=request.covariates,
            method=request.method,
            n_folds=request.n_folds,
        )
        ranked_df = rank_farms_by_uplift(
            cate,
            intervention_cost_usd_per_farm=request.intervention_cost_usd_per_farm,
            cocoa_price_usd=request.cocoa_price_usd,
            farm_areas_ha=df[request.farm_area_col],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    farm_id = ranked_df["farm_id"] if "farm_id" in ranked_df.columns else pd.Series([None] * len(ranked_df), index=ranked_df.index)
    ranked = []
    for idx in ranked_df.index:
        ranked.append(
            {
                "farm_id": farm_id.loc[idx],
                "net_uplift_usd": float(ranked_df.loc[idx, "net_uplift_usd"]),
                "gross_uplift_usd": float(ranked_df.loc[idx, "gross_uplift_usd"]),
                "avoided_loss_tonnes": float(ranked_df.loc[idx, "avoided_loss_tonnes"]),
                "tau_hat_tonnes_per_ha": float(ranked_df.loc[idx, "tau_hat_tonnes_per_ha"]),
                "se": float(ranked_df.loc[idx, "se"]),
            }
        )

    return RankInterventionsResponse(method=request.method, n=int(len(df)), ranked=ranked)


@app.post(
    "/compliance/dds",
    response_model=ComplianceDdsResponse,
    summary="Generate EUDR due diligence statement (cocoa)",
)
def compliance_dds_endpoint(request: ComplianceDdsRequest) -> ComplianceDdsResponse:
    """
    Validate plot geolocation (Art. 2(28)), optional deforestation screening (Art. 3),
    country risk (Art. 29), and return a due diligence statement with risk score (Art. 10).
    """
    validation = validate_geolocation(request.plot)
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail={"geolocation_errors": validation.errors})

    country_risk = assess_country_risk(request.plot.country)

    if request.use_gee_deforestation_check:
        deforestation = check_deforestation_free(request.plot)
    else:
        deforestation = DeforestationResult(
            is_deforestation_free=True,
            loss_pixels=0,
            loss_area_ha=0.0,
            notes=["GEE deforestation check skipped (use_gee_deforestation_check=false)"],
        )

    dds = generate_dds(
        request.plot,
        request.operator,
        request.product,
        buyer_name=request.buyer_name,
        supplier_name=request.supplier_name,
        deforestation_result=deforestation,
        country_risk=country_risk,
        supply_chain_complexity=request.supply_chain_complexity,
    )

    return ComplianceDdsResponse(
        dds=dds,
        dds_json=dds.to_json(),
        dds_csv=dds.to_eu_csv(),
        risk_score=dds.risk_score,
        geolocation_valid=dds.geolocation_valid,
        validation_errors=dds.validation_errors,
    )
