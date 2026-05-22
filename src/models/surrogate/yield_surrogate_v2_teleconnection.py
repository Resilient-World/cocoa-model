"""
YieldSurrogateV2 with teleconnection GNN bias (ENSO / Atl3 / IOD).
"""

from __future__ import annotations

import structlog

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from models.features.teleconnection_gnn import TeleconnectionFeatures, TeleconnectionGNN
from models.surrogate.yield_surrogate import MechanisticTraces
from models.surrogate.yield_surrogate_v2 import YieldSurrogateV2

log = structlog.get_logger(__name__)


class YieldSurrogateV2Teleconnection(nn.Module):
    """
    Composite yield engine: PAPE+GRU surrogate plus teleconnection ``delta_y``.

    MC dropout runs on the surrogate branch inside each forward call; GNN bias is
    deterministic per farm/year teleconnection features.
    """

    def __init__(
        self,
        surrogate: YieldSurrogateV2,
        teleconnection: TeleconnectionGNN | None = None,
    ) -> None:
        super().__init__()
        self.surrogate = surrogate
        self.teleconnection = teleconnection or TeleconnectionGNN()

    def train(self, mode: bool = True) -> YieldSurrogateV2Teleconnection:
        self.surrogate.train(mode)
        self.teleconnection.train(mode)
        return self

    def eval(self) -> YieldSurrogateV2Teleconnection:
        self.surrogate.eval()
        self.teleconnection.eval()
        return self

    def _teleconnection_delta(
        self,
        teleconnection: TeleconnectionFeatures | dict[str, Any],
        *,
        region_id: Tensor | None,
        lat: float | Tensor,
        lon: float | Tensor,
    ) -> Tensor:
        if isinstance(teleconnection, TeleconnectionFeatures):
            delta = self.teleconnection(teleconnection)
            return delta.unsqueeze(0) if delta.ndim == 0 else delta

        nino = teleconnection["nino34"]
        if hasattr(nino, "ndim") and getattr(nino, "ndim", 0) > 1:
            batch_size = int(nino.shape[0])
            lats = lat if isinstance(lat, Tensor) else torch.full((batch_size,), float(lat))
            lons = lon if isinstance(lon, Tensor) else torch.full((batch_size,), float(lon))
            if not isinstance(lats, Tensor):
                lats = torch.tensor(lats)
            if not isinstance(lons, Tensor):
                lons = torch.tensor(lons)
            features = []
            for i in range(batch_size):
                rid = int(region_id[i].item()) if region_id is not None else 0
                features.append(
                    TeleconnectionFeatures.from_dict(
                        {
                            "nino34": _slice_index(teleconnection["nino34"], i),
                            "atl3": _slice_index(teleconnection["atl3"], i),
                            "iod": _slice_index(teleconnection["iod"], i),
                        },
                        region_id=rid,
                        lat=float(lats[i].item()),
                        lon=float(lons[i].item()),
                    )
                )
            return self.teleconnection(features)

        rid = int(region_id.view(-1)[0].item()) if region_id is not None else 0
        tf = TeleconnectionFeatures.from_dict(
            teleconnection,
            region_id=rid,
            lat=float(lat if not isinstance(lat, Tensor) else lat.item()),
            lon=float(lon if not isinstance(lon, Tensor) else lon.item()),
        )
        delta = self.teleconnection(tf)
        return delta.unsqueeze(0) if delta.ndim == 0 else delta

    def forward(
        self,
        climate: Tensor,
        static: Tensor,
        region_id: Tensor | None = None,
        teleconnection: TeleconnectionFeatures | dict[str, Any] | None = None,
        *,
        doy: Tensor | None = None,
        lat: float | Tensor = 6.0,
        lon: float | Tensor = -2.0,
    ) -> Tensor:
        y_base = self.surrogate(climate, static, region_id, doy=doy)
        if teleconnection is None:
            return y_base
        delta = self._teleconnection_delta(
            teleconnection,
            region_id=region_id,
            lat=lat,
            lon=lon,
        )
        return y_base + delta.to(y_base.dtype).to(y_base.device)

    def forward_with_traces(
        self,
        climate: Tensor,
        static: Tensor,
        region_id: Tensor | None = None,
        teleconnection: TeleconnectionFeatures | dict[str, Any] | None = None,
        *,
        doy: Tensor | None = None,
        lat: float | Tensor = 6.0,
        lon: float | Tensor = -2.0,
    ) -> tuple[Tensor, MechanisticTraces]:
        y_base, traces = self.surrogate.forward_with_traces(
            climate, static, region_id, doy=doy
        )
        if teleconnection is None:
            return y_base, traces
        delta = self._teleconnection_delta(
            teleconnection,
            region_id=region_id,
            lat=lat,
            lon=lon,
        )
        y = y_base + delta.to(y_base.dtype).to(y_base.device)
        return y, traces

    @classmethod
    def from_checkpoints(
        cls,
        surrogate_path: str | Path,
        teleconnection_path: str | Path | None = None,
        *,
        map_location: str | torch.device = "cpu",
    ) -> YieldSurrogateV2Teleconnection:
        """Load surrogate (v2) and optional teleconnection head weights."""
        sur = YieldSurrogateV2.from_checkpoint(surrogate_path, map_location=map_location)
        gnn = TeleconnectionGNN()
        model = cls(sur, gnn)
        if teleconnection_path is not None and Path(teleconnection_path).is_file():
            blob = torch.load(teleconnection_path, map_location=map_location, weights_only=False)
            if isinstance(blob, dict) and "state_dict" in blob:
                model.load_state_dict(blob["state_dict"], strict=False)
            elif isinstance(blob, dict):
                model.teleconnection.load_state_dict(blob, strict=False)
            log.info("Loaded teleconnection weights from %s", teleconnection_path)
        model.eval()
        return model


def _slice_index(arr: Any, i: int) -> Any:
    if hasattr(arr, "cpu"):
        return arr[i].cpu().numpy()
    return np.asarray(arr)[i]


def freeze_surrogate_except_pape(model: YieldSurrogateV2Teleconnection) -> None:
    """Freeze all surrogate weights except PAPE (for teleconnection finetune)."""
    for name, param in model.surrogate.named_parameters():
        param.requires_grad = name.startswith("pape.")
    for param in model.teleconnection.parameters():
        param.requires_grad = True


__all__ = [
    "YieldSurrogateV2Teleconnection",
    "freeze_surrogate_except_pape",
]
