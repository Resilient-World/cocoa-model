"""FastAPI entrypoint for the Avoided Loss simulation service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException

from api.config import APISettings
from api.feature_resolver import build_resolver_from_settings
from api.model_loader import load_yield_model
from api.schemas import (
    ComplianceDdsRequest,
    ComplianceDdsResponse,
    SimulateInterventionRequest,
    SimulateInterventionResponse,
)
from api.simulation import simulate_intervention
from compliance.eudr import (
    DeforestationResult,
    assess_country_risk,
    check_deforestation_free,
    generate_dds,
    validate_geolocation,
)
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
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
