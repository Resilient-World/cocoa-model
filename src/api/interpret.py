"""Auth-gated TCAV interpretability endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from analysis.tcav import CONCEPT_IDS, TCAVResult, tcav_scores
from api.config import APISettings

router = APIRouter(prefix="", tags=["interpret"])


class InterpretRequest(BaseModel):
    farm_location_lat: float = Field(..., ge=-90, le=90)
    farm_location_lon: float = Field(..., ge=-180, le=180)
    intervention_type: str = "shade_trees"
    concepts: list[str] | None = None


class InterpretConceptScore(BaseModel):
    concept: str
    score: float
    p_value: float
    n_concept: int
    n_random: int


class InterpretResponse(BaseModel):
    intervention_type: str
    concepts: list[InterpretConceptScore]


def _verify_interpret_token(request: Request) -> None:
    settings: APISettings = request.app.state.settings
    token = getattr(settings, "interpret_auth_token", None)
    if not token:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/interpret", response_model=InterpretResponse)
def interpret_endpoint(
    body: InterpretRequest,
    request: Request,
    _: None = Depends(_verify_interpret_token),
) -> InterpretResponse:
    if not getattr(request.app.state.settings, "interpret_enabled", False):
        raise HTTPException(status_code=503, detail="INTERPRET_ENABLED=false")
    import torch

    model = request.app.state.yield_model
    resolver = request.app.state.feature_resolver
    lat, lon = body.farm_location_lat, body.farm_location_lon
    climate = resolver.resolve_climate(lat, lon, 2023)
    static = resolver.resolve_static_with_galileo(lat, lon, 2023)
    concepts = body.concepts or list(CONCEPT_IDS)
    if not hasattr(model, "forward_with_activations"):
        raise HTTPException(status_code=501, detail="Model does not support TCAV activations")
    results = tcav_scores(
        model,
        climate=climate,
        static=static,
        concepts=concepts,
    )
    return InterpretResponse(
        intervention_type=body.intervention_type,
        concepts=[
            InterpretConceptScore(
                concept=r.concept,
                score=r.score,
                p_value=r.p_value,
                n_concept=r.n_concept,
                n_random=r.n_random,
            )
            for r in results
        ],
    )
