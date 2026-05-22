"""TCAV scores over yield surrogate hidden activations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

CONCEPT_IDS = (
    "drought_year",
    "shade_present",
    "high_rainfall",
    "cssvd_zone",
    "mid_crop_season",
)


@dataclass
class TCAVResult:
    concept: str
    score: float
    p_value: float
    n_concept: int
    n_random: int


def _concept_mask(concept: str, climate: Tensor, static: Tensor) -> Tensor:
    """Boolean mask over batch for concept membership."""
    b = climate.shape[0]
    if concept == "drought_year":
        precip = climate[..., 3] if climate.shape[-1] > 3 else climate.mean(dim=-1)
        return precip.mean(dim=1) < precip.mean()
    if concept == "shade_present":
        return static[..., 5] > 0.3 if static.shape[-1] > 5 else torch.zeros(b, dtype=torch.bool)
    if concept == "high_rainfall":
        precip = climate[..., 3] if climate.shape[-1] > 3 else climate.mean(dim=-1)
        return precip.mean(dim=1) > precip.quantile(0.75)
    if concept == "cssvd_zone":
        return static[..., 0] > 0.5 if static.shape[-1] > 0 else torch.zeros(b, dtype=torch.bool)
    if concept == "mid_crop_season":
        doy = torch.arange(climate.shape[1], device=climate.device).float()
        mid = (doy > 90) & (doy < 180)
        tmean = climate[:, mid, 2] if climate.shape[-1] > 2 else climate[:, mid].mean(-1)
        return tmean.mean(dim=1) > tmean.mean()
    raise ValueError(f"Unknown concept: {concept}")


def tcav_scores(
    model: torch.nn.Module,
    *,
    climate: Tensor,
    static: Tensor,
    concepts: Sequence[str] = CONCEPT_IDS,
    n_random: int = 50,
    random_state: int = 42,
) -> list[TCAVResult]:
    """
    Compute TCAV scores with two-sided t-test vs random concept baselines.
    """
    model.eval()
    climate = climate.detach()
    static = static.detach()
    climate.requires_grad_(True)
    y, hidden = model.forward_with_activations(climate, static)
    grad = torch.autograd.grad(y.sum(), hidden, retain_graph=True)[0]
    rng = np.random.default_rng(random_state)
    results: list[TCAVResult] = []
    for concept in concepts:
        mask = _concept_mask(concept, climate.detach(), static)
        if mask.sum() < 2:
            results.append(TCAVResult(concept, 0.0, 1.0, int(mask.sum()), n_random))
            continue
        concept_dir = grad[mask].mean(dim=0)
        concept_dir = concept_dir / (concept_dir.norm() + 1e-8)
        concept_sens = (grad[mask] @ concept_dir).detach().cpu().numpy()
        random_sens = []
        for _ in range(n_random):
            rnd = torch.randn_like(concept_dir)
            rnd = rnd / (rnd.norm() + 1e-8)
            random_sens.append(float((grad @ rnd).mean().item()))
        score = float((concept_sens > 0).mean())
        from scipy import stats

        tstat, pval = stats.ttest_ind(concept_sens, random_sens, equal_var=False)
        results.append(
            TCAVResult(
                concept=concept,
                score=score,
                p_value=float(pval) if pval == pval else 1.0,
                n_concept=int(mask.sum()),
                n_random=n_random,
            )
        )
    return results
