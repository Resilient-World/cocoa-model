"""FastAPI entrypoint for the Avoided Loss simulation service."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from analysis.heterogeneity import estimate_cate
from analysis.policy_targeting import (
    learn_policy_forest,
    learn_policy_tree,
    rank_farms_by_uplift,
    render_policy_rules,
    render_policy_rules_from_forest,
)
from api import metrics as prom_metrics
from api import telemetry
from api.config import APISettings
from api.cqr_loader import load_cqr_bundle
from api.drift_monitoring import get_drift_status_for_stratum
from api.eudr import router as eudr_router
from api.feature_resolver import build_resolver_from_settings
from api.interpret import router as interpret_router
from api.model_loader import load_casej_model, load_yield_model
from api.observability_middleware import register_observability_middleware
from api.online_conformal_store import build_store_from_settings
from api.schemas import (
    ComplianceDdsRequest,
    ComplianceDdsResponse,
    DriftStatus,
    ExposureCanopyRequest,
    ExposureCanopyResponse,
    LearnPolicyRulesRequest,
    LearnPolicyRulesResponse,
    PolicyRule,
    PolicyRulebook,
    PriceParametricRequest,
    PriceParametricResponse,
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
from common.logging import configure_logging
from compliance.eudr import (
    DeforestationResult,
    assess_country_risk,
    check_deforestation_free,
    generate_dds,
    validate_geolocation,
)
from finance.parametric_insurance import price_parametric_trigger
from models.conformal import load_conformal_if_exists
from models.yield_surrogate import YieldSurrogateModel
from monitoring.drift_store import build_drift_store_from_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        json=os.environ.get("LOG_JSON", "true").lower() in ("1", "true", "yes"),
    )
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
    app.state.scenario_conformal_store = build_store_from_settings(settings)
    app.state.drift_store = build_drift_store_from_settings(settings)

    if settings.otel_enabled:
        try:
            from importlib.metadata import version as pkg_version

            svc_ver = settings.otel_service_version or pkg_version("resilient-cocoa-model")
        except Exception:
            svc_ver = settings.otel_service_version
        telemetry.configure_tracing(
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
            service_name=settings.otel_service_name,
            service_version=svc_ver,
            environment=settings.otel_deployment_environment,
        )

    prom_metrics.setup_metrics(app, settings)
    register_observability_middleware(app)
    if settings.otel_enabled:
        telemetry.instrument_fastapi(app)

    yield


app = FastAPI(
    title="Resilient Cocoa Model API",
    description="Geospatial ML inference and intervention simulation",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(eudr_router)
app.include_router(interpret_router)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    endpoint = request.url.path.strip("/") or "root"
    if exc.status_code >= 500:
        prom_metrics.inc_simulation_error("http_5xx", endpoint)
    elif exc.status_code >= 400:
        prom_metrics.inc_simulation_error("http_4xx", endpoint)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health() -> dict[str, str]:
    """
    Liveness probe for load balancers and CI smoke tests.

    Returns
    -------
    dict
        ``{"status": "ok"}`` when the process is up (no model inference).
    """
    return {"status": "ok"}


@app.post(
    "/exposure-canopy",
    response_model=ExposureCanopyResponse,
    summary="Sample GEDI/ICESat-2 canopy structure for a farm point",
)
def exposure_canopy_endpoint(request: ExposureCanopyRequest) -> ExposureCanopyResponse:
    """Return canopy height, cover, biomass, and source attribution for one farm."""
    loc = request.farm_location
    sample = app.state.feature_resolver.resolve_canopy(loc.lat, loc.lon, request.year)
    return ExposureCanopyResponse(**sample.as_dict())


@app.post(
    "/price-parametric",
    response_model=PriceParametricResponse,
    summary="Price a farm-level parametric yield trigger",
)
def price_parametric_endpoint(request: PriceParametricRequest) -> PriceParametricResponse:
    """Return fair and DVDS-loaded premium plus basis-risk metrics."""
    loc = request.farm_location
    resolver = app.state.feature_resolver
    climate = resolver.resolve_climate(loc.lat, loc.lon, 2023)
    precip = climate[..., 3].detach().cpu().numpy().reshape(-1)
    annual_precip = float(np.sum(precip))
    drought_index = np.clip((1200.0 - annual_precip) / 1200.0, -0.4, 0.8)
    seed = abs(hash((round(loc.lat, 3), round(loc.lon, 3), request.scenario))) % (2**32)
    rng = np.random.default_rng(seed)
    mean_yield = max(0.2, request.strike_t_per_ha * (1.08 - 0.35 * drought_index))
    samples = rng.normal(mean_yield, max(0.08, 0.18 * mean_yield), 256)
    samples = np.clip(samples, 0.05, None)
    width = max(0.1, float(np.std(samples)) * 3.29)
    report = price_parametric_trigger(
        request.strike_t_per_ha,
        samples,
        {"lower": mean_yield - width / 2.0, "upper": mean_yield + width / 2.0},
        farm_size_ha=request.farm_size_ha,
        price_usd_per_t=request.cocoa_price_usd,
    )
    payload = report.as_dict()
    payload["scenario"] = request.scenario
    payload["coverage_horizon_years"] = request.coverage_horizon_years
    return PriceParametricResponse(**payload)


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
            settings=settings,
        )
    except ValueError as exc:
        prom_metrics.inc_simulation_error("ValueError", "simulate-intervention")
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
    settings: APISettings = app.state.settings
    if settings.scenario_yield_backend == "casej":
        model = app.state.casej_model
    else:
        model = app.state.yield_model

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
            settings=settings,
            cqr_model=app.state.cqr_model,
            cqr_calibrator=app.state.cqr_calibrator,
            scenario_conformal_store=app.state.scenario_conformal_store,
            drift_store=app.state.drift_store,
        )
    except ValueError as exc:
        msg = str(exc)
        if "CorrDiff cache" in msg or "CorrDiff requires" in msg or "CorrDiffCMIP6" in msg:
            raise HTTPException(status_code=503, detail=msg) from exc
        if "AURORA_COMMERCIAL_OK" in msg:
            raise HTTPException(status_code=422, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc


@app.get(
    "/drift-status",
    response_model=DriftStatus,
    summary="Current WCTM drift state for a conformal stratum",
)
def drift_status_endpoint(stratum: str) -> DriftStatus:
    """
    Return persisted WCTM log-martingale, alarm flag, and diagnosis for dashboards.

    ``stratum`` must match ``{scenario}:{horizon_year}:{region}`` (e.g. ``ssp245:2050:ghana``).
    """
    try:
        return get_drift_status_for_stratum(
            stratum,
            drift_store=app.state.drift_store,
            conformal_store=app.state.scenario_conformal_store,
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
    prom_metrics.inc_policy_endpoint("rank-interventions")
    df = pd.DataFrame(request.rows)
    covariates = list(request.covariates)
    if request.condition_on_canopy and request.treatment == "shade_trees":
        resolver = app.state.feature_resolver
        if "canopy_height_m" not in df.columns or "agb_mg_ha" not in df.columns:
            for idx, row in df.iterrows():
                lat = row.get("lat", row.get("latitude"))
                lon = row.get("lon", row.get("longitude"))
                year = int(row.get("year", 2023))
                if lat is None or lon is None:
                    continue
                sample = resolver.resolve_canopy(float(lat), float(lon), year)
                if "canopy_height_m" not in df.columns:
                    df.loc[idx, "canopy_height_m"] = sample.canopy_height_m
                if "agb_mg_ha" not in df.columns:
                    df.loc[idx, "agb_mg_ha"] = sample.agb_mg_ha
        if "canopy_height_m" in df.columns:
            df["canopy_percentile"] = df["canopy_height_m"].rank(pct=True).fillna(0.5)
            for name in ("canopy_height_m", "canopy_percentile", "agb_mg_ha"):
                if name in df.columns and name not in covariates:
                    covariates.append(name)
    if request.farm_area_col not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Missing farm area column '{request.farm_area_col}'"
        )
    try:
        cate = estimate_cate(
            df,
            outcome=request.outcome,
            treatment=request.treatment,
            covariates=covariates,
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

    farm_id = (
        ranked_df["farm_id"]
        if "farm_id" in ranked_df.columns
        else pd.Series([None] * len(ranked_df), index=ranked_df.index)
    )
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

    return RankInterventionsResponse(method=request.method, n=len(df), ranked=ranked)


def _rulebook_from_tree_result(
    result: Any,
    *,
    method: str,
    rules_text: list[str],
    n_samples: int,
) -> PolicyRulebook:
    rules: list[PolicyRule] = []
    for i, row in result.leaf_summary.iterrows():
        text = str(row["rule_text"]) if pd.notna(row["rule_text"]) and row["rule_text"] else ""
        rules.append(
            PolicyRule(
                rule_id=int(i),
                rule_text=text,
                leaf_id=int(row["leaf_id"]),
                n_units=int(row["n_units"]),
                treat_fraction=float(row["treat_fraction"]),
                expected_uplift=float(row["expected_uplift"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
            )
        )
    if not rules and rules_text:
        for rid, text in enumerate(rules_text):
            rules.append(
                PolicyRule(
                    rule_id=rid,
                    rule_text=text,
                    leaf_id=rid,
                    n_units=0,
                    treat_fraction=0.0,
                    expected_uplift=0.0,
                    ci_low=0.0,
                    ci_high=0.0,
                )
            )
    return PolicyRulebook(
        method=method,  # type: ignore[arg-type]
        feature_names=result.feature_names,
        treatment_names=result.treatment_names,
        rules=rules,
        policy_value_estimate=float(result.policy_value_estimate),
        policy_value_ci_low=float(result.policy_value_ci[0]),
        policy_value_ci_high=float(result.policy_value_ci[1]),
        greedy_policy_value=result.greedy_policy_value,
        cost_aware=bool(result.cost_aware),
        n_samples=n_samples,
    )


@app.post(
    "/learn-policy-rules",
    response_model=LearnPolicyRulesResponse,
    summary="Learn interpretable DR policy targeting rules from a farm panel",
)
def learn_policy_rules_endpoint(request: LearnPolicyRulesRequest) -> LearnPolicyRulesResponse:
    """
    Fit an honest doubly-robust policy tree or forest and return regulator-readable
    if-then rules with leaf-level uplift statistics.
    """
    df = pd.DataFrame(request.rows)
    if len(df) < request.min_samples_leaf * 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {request.min_samples_leaf * 2} rows for policy learning",
        )
    try:
        if request.learner == "forest":
            result = learn_policy_forest(
                df,
                treatment_col=request.treatment,
                outcome_col=request.outcome,
                covariate_cols=request.covariates,
                max_depth=request.max_depth,
                min_samples_leaf=request.min_samples_leaf,
                cost_col=request.cost_col,
                n_estimators=request.n_estimators,
                n_folds=request.n_folds,
                random_state=request.random_state,
                n_bootstrap=request.n_bootstrap,
                intervention_cost_usd_per_farm=request.intervention_cost_usd_per_farm,
                budget=request.budget,
                recommended_treatment_label=request.recommended_treatment_label,
                cate_method=request.cate_method,
            )
            rules_text = render_policy_rules_from_forest(
                result,
                recommended_treatment_label=request.recommended_treatment_label,
            )
            rulebook = _rulebook_from_tree_result(
                result,
                method="policy_forest",
                rules_text=rules_text,
                n_samples=len(df),
            )
        else:
            result = learn_policy_tree(
                df,
                treatment_col=request.treatment,
                outcome_col=request.outcome,
                covariate_cols=request.covariates,
                max_depth=request.max_depth,
                min_samples_leaf=request.min_samples_leaf,
                cost_col=request.cost_col,
                n_folds=request.n_folds,
                random_state=request.random_state,
                n_bootstrap=request.n_bootstrap,
                intervention_cost_usd_per_farm=request.intervention_cost_usd_per_farm,
                budget=request.budget,
                recommended_treatment_label=request.recommended_treatment_label,
                cate_method=request.cate_method,
            )
            rules_text = render_policy_rules(
                result,
                recommended_treatment_label=request.recommended_treatment_label,
            )
            rulebook = _rulebook_from_tree_result(
                result,
                method="policy_tree",
                rules_text=rules_text,
                n_samples=len(df),
            )
    except (ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LearnPolicyRulesResponse(rulebook=rulebook)


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
