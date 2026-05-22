"""
YieldSurrogateV2: v1 mechanistic + GRU surrogate with PAPE on the climate branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import torch
from torch import Tensor

from data.cocoa_exposure import REGIONS
from finance.pricing import infer_country_code
from models.features.pape import (
    REGION_KEYS,
    PhenologyAwarePositionalEncoding,
    region_to_id,
)
from models.io.checkpoint_migration import is_v1_static_checkpoint, migrate_v1_static_to_v2
from models.surrogate.yield_surrogate import (
    N_STATIC_SITE,
    MechanisticTraces,
    YieldSurrogateModel,
)

log = structlog.get_logger(__name__)

_ISO3_TO_REGION: dict[str, str] = {
    "GHA": "ghana",
    "CIV": "civ",
    "CMR": "cameroon",
    "NGA": "nigeria",
    "IDN": "indonesia",
    "ECU": "ecuador",
    "PER": "peru",
    "COL": "colombia",
}


def region_id_from_country_code(country_code: str) -> int:
    """Map ISO3 producer code to phenology region id."""
    key = _ISO3_TO_REGION.get(country_code.strip().upper())
    if key is None:
        raise KeyError(
            f"Unknown country_code {country_code!r}; expected one of {sorted(_ISO3_TO_REGION)}"
        )
    return region_to_id(key)


def region_id_from_latlon(lat: float, lon: float) -> int:
    """Pick the :data:`~data.cocoa_exposure.REGIONS` bbox containing ``(lat, lon)``."""
    matches: list[str] = []
    for key, preset in REGIONS.items():
        if preset.west <= lon <= preset.east and preset.south <= lat <= preset.north:
            matches.append(key)
    if len(matches) == 1:
        return region_to_id(matches[0])
    if not matches:
        return region_id_from_country_code(infer_country_code(lat, lon))
    return region_to_id(matches[0])


def _infer_dims_from_state(state: dict[str, Any]) -> tuple[int, int, int]:
    """Return ``(climate_features, site_static, galileo_dim)`` from checkpoint keys."""
    gru_w = state.get("climate_gru.weight_ih_l0")
    if not isinstance(gru_w, Tensor):
        climate_features = 11
    else:
        climate_features = int(gru_w.shape[1])

    static_w = state.get("static_mlp.0.weight")
    if not isinstance(static_w, Tensor):
        return climate_features, N_STATIC_SITE, 0
    static_in = int(static_w.shape[1])
    galileo_dim = max(0, static_in - N_STATIC_SITE)
    return climate_features, N_STATIC_SITE, galileo_dim


class YieldSurrogateV2(YieldSurrogateModel):
    """Yield surrogate with PAPE applied before the climate GRU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pape = PhenologyAwarePositionalEncoding(
            self.climate_features,
            n_regions=len(REGION_KEYS),
        )

    def _apply_pape(
        self,
        climate: Tensor,
        region_id: Tensor | None,
        *,
        doy: Tensor | None = None,
    ) -> Tensor:
        if region_id is None:
            region_id = torch.zeros(climate.shape[0], dtype=torch.long, device=climate.device)
        return self.pape(climate, region_id, doy=doy)

    def forward_with_traces(
        self,
        climate: Tensor,
        static: Tensor,
        region_id: Tensor | None = None,
        *,
        doy: Tensor | None = None,
    ) -> tuple[Tensor, MechanisticTraces]:
        self._validate_inputs(climate, static)
        climate_full = self._pad_climate(climate)
        climate_enc = self._apply_pape(climate_full, region_id, doy=doy)
        traces = self.mechanistic(climate_full, static)
        residual = self._encode(climate_enc, static, traces)
        y = traces["y_mech"] + residual
        return y, traces

    def forward(
        self,
        climate: Tensor,
        static: Tensor,
        region_id: Tensor | None = None,
        *,
        doy: Tensor | None = None,
    ) -> Tensor:
        y, _ = self.forward_with_traces(climate, static, region_id, doy=doy)
        return y

    @classmethod
    def from_v1_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> YieldSurrogateV2:
        """
        Load a v1 (or migrated v2-static) checkpoint into v2 with PAPE at identity.

        PAPE weights stay zero-initialized when absent from the checkpoint.
        """
        ckpt_path = Path(path)
        blob = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        if not isinstance(state, dict):
            raise TypeError(f"Expected state dict in {ckpt_path}")

        if is_v1_static_checkpoint(state):
            state = migrate_v1_static_to_v2(state)

        climate_features, site_static, galileo_dim = _infer_dims_from_state(state)
        model = cls(
            climate_features=climate_features,
            static_features=site_static,
            galileo_dim=galileo_dim,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        pape_missing = [k for k in missing if k.startswith("pape.")]
        other_missing = [k for k in missing if not k.startswith("pape.")]
        if other_missing:
            log.warning("Checkpoint missing non-PAPE keys: %s", other_missing[:8])
        if unexpected:
            log.warning("Unexpected checkpoint keys: %s", unexpected[:8])
        if pape_missing:
            log.info("PAPE keys absent (%d); using zero-init identity", len(pape_missing))

        model.eval()
        with torch.no_grad():
            climate = torch.zeros(1, model.sequence_length, model.climate_features)
            static = torch.zeros(1, model.static_features)
            rid = torch.zeros(1, dtype=torch.long)
            delta = model.pape.delta(climate, rid)
            max_delta = float(delta.abs().max().item())
            if max_delta > 1e-6:
                raise RuntimeError(f"PAPE not at identity after v1 load (max |delta|={max_delta})")
        return model

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> YieldSurrogateV2:
        """Load a v2 checkpoint (config + state_dict) or fall back to v1 migration."""
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        blob = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and blob.get("version") == "v2" and "config" in blob:
            cfg = blob["config"]
            model = cls(
                sequence_length=int(cfg.get("sequence_length", 365)),
                climate_features=int(cfg.get("climate_features", 11)),
                galileo_dim=int(cfg.get("galileo_dim", 0)),
            )
            state = blob["state_dict"]
            if is_v1_static_checkpoint(state):
                state = migrate_v1_static_to_v2(state)
            model.load_state_dict(state, strict=False)
            model.eval()
            return model
        return cls.from_v1_checkpoint(ckpt_path, map_location=map_location)


__all__ = [
    "YieldSurrogateV2",
    "region_id_from_country_code",
    "region_id_from_latlon",
]
