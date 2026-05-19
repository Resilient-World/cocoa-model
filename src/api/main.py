"""FastAPI entrypoint for the Avoided Loss simulation service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException

from api.config import APISettings
from api.model_loader import load_yield_model
from api.schemas import SimulateInterventionRequest, SimulateInterventionResponse
from api.simulation import simulate_intervention
from models.yield_surrogate import YieldSurrogateModel


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = APISettings()
    app.state.settings = settings
    app.state.yield_model = load_yield_model(settings.model_checkpoint_path)
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
    Mock climate/soil retrieval, run yield surrogate inference for counterfactual
    and factual scenarios, and return avoided loss with a 90% confidence interval.
    """
    model: YieldSurrogateModel = app.state.yield_model
    settings: APISettings = app.state.settings

    try:
        return simulate_intervention(
            request,
            model,
            num_samples=settings.mc_num_samples,
            yield_blend_weight=settings.yield_blend_weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
