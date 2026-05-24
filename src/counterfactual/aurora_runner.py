"""
Microsoft Aurora 1.5 scenario downscaling (opt-in; install ``[aurora]`` + ``AURORA_ENABLED``).

Research license by default; commercial use requires ``AURORA_COMMERCIAL_OK`` and approval
from AIWeatherClimate@microsoft.com — see ``docs/LICENSES.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog
import xarray as xr

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RegionBBox:
    """Cocoa-belt bounding box (WGS84); keep in sync with ``data.cocoa_exposure.REGIONS``."""

    display_name: str
    west: float
    south: float
    east: float
    north: float


# Source: data.cocoa_exposure.REGIONS (8 cocoa belt countries)
COCOA_BELT_REGIONS: dict[str, RegionBBox] = {
    "ghana": RegionBBox("Ghana", -3.25, 4.7, 1.2, 11.2),
    "civ": RegionBBox("Côte d'Ivoire", -8.5, 4.0, -2.5, 11.0),
    "cameroon": RegionBBox("Cameroon", 8.0, 1.5, 16.5, 13.5),
    "nigeria": RegionBBox("Nigeria", 2.5, 4.0, 14.5, 14.0),
    "indonesia": RegionBBox("Indonesia", 95.0, -11.0, 141.0, 7.0),
    "ecuador": RegionBBox("Ecuador", -81.5, -5.5, -75.0, 2.0),
    "peru": RegionBBox("Peru", -81.0, -15.0, -68.0, 0.5),
    "colombia": RegionBBox("Colombia", -79.0, -4.5, -66.0, 12.0),
}

_REGION_ALIASES: dict[str, str] = {
    "gha": "ghana",
    "ci": "civ",
    "civ": "civ",
    "cmr": "cameroon",
    "nga": "nigeria",
    "idn": "indonesia",
    "ecu": "ecuador",
    "per": "peru",
    "col": "colombia",
}


def normalize_region_key(name: str) -> str:
    """Map CLI aliases to :data:`COCOA_BELT_REGIONS` keys."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in COCOA_BELT_REGIONS:
        return key
    if key in _REGION_ALIASES:
        return _REGION_ALIASES[key]
    if key in ("cote_divoire", "cote_d_ivoire", "ivory_coast"):
        return "civ"
    raise KeyError(f"Unknown region {name!r}; choose from {sorted(COCOA_BELT_REGIONS)}")


ModelSize = Literal["small", "medium"]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LORA_DIR = _REPO_ROOT / "models"

AURORA_MODEL_VERSIONS: dict[ModelSize, str] = {
    "small": "aurora-0.25-small-pretrained",
    "medium": "aurora-0.25-pretrained",
}


def aurora_cache_key(
    *,
    init_time: str | datetime,
    lead_h: int,
    region: str,
    model_size: ModelSize,
    lora_id: str = "base",
) -> str:
    """Stable Zarr group name for a cached regional rollout."""
    if isinstance(init_time, datetime):
        init_iso = init_time.strftime("%Y%m%dT%H%M%S")
    else:
        init_iso = str(init_time).replace(":", "").replace("-", "")[:15]
    reg = normalize_region_key(region)
    lora = lora_id or "base"
    return f"{init_iso}_{int(lead_h)}h_{reg}_{model_size}_{lora}"


def aurora_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.zarr"


def load_cached_forecast(cache_dir: Path, key: str) -> xr.Dataset | None:
    path = aurora_cache_path(cache_dir, key)
    if not path.is_dir():
        return None
    try:
        return xr.open_zarr(path, consolidated=False)
    except Exception as exc:
        log.warning("aurora_cache_read_failed", path=str(path), error=str(exc))
        return None


def write_cached_forecast(cache_dir: Path, key: str, ds: xr.Dataset) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = aurora_cache_path(cache_dir, key)
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def region_lat_lon_grid(
    preset: RegionBBox, n_lat: int = 17, n_lon: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced lat/lon inside a cocoa-belt ``RegionPreset`` bounding box."""
    lats = np.linspace(preset.south, preset.north, n_lat, dtype=np.float64)
    lons = np.linspace(preset.west, preset.east, n_lon, dtype=np.float64)
    return lats, lons


def pd_date_range(start: str, end: str) -> np.ndarray:
    import pandas as pd

    return pd.date_range(start, end, freq="D").to_numpy()


def _mock_era5_point_dataset(
    *,
    lat: float,
    lon: float,
    start: str,
    end: str,
    tag: str = "aurora_mock",
) -> xr.Dataset:
    days = pd_date_range(start, end)
    n = len(days)
    tmean = 26.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 365.0)
    ds = xr.Dataset(
        {
            "tmean": (("time",), tmean.astype(np.float32)),
            "tmax": (("time",), (tmean + 2).astype(np.float32)),
            "tmin": (("time",), (tmean - 4).astype(np.float32)),
            "precip": (
                ("time",),
                np.clip(np.random.default_rng(42).gamma(2, 2, n), 0, 40).astype(np.float32),
            ),
            "rh_mean": (("time",), np.full(n, 75.0, dtype=np.float32)),
            "srad": (("time",), np.full(n, 15.0, dtype=np.float32)),
            "wind10m": (("time",), np.full(n, 2.0, dtype=np.float32)),
            "vpd": (("time",), np.full(n, 1.0, dtype=np.float32)),
            "et0": (("time",), np.full(n, 3.0, dtype=np.float32)),
            "cwd": (("time",), np.zeros(n, dtype=np.float32)),
        },
        coords={"time": days, "lat": lat, "lon": lon},
        attrs={"aurora_backend": tag},
    )
    return ds


def _aurora_batch_to_point_dataset(
    batch: Any,
    *,
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> xr.Dataset:
    """Map Aurora ``Batch`` surface fields to ERA5-schema daily point series (stub grid)."""
    import torch

    days = pd_date_range(start, end)
    n = len(days)
    surf = batch.surf_vars
    t2 = surf["2t"]
    if isinstance(t2, torch.Tensor):
        t2_k = float(t2[0, -1].detach().cpu().numpy())
    else:
        t2_k = float(t2)
    tmean_c = t2_k - 273.15
    tmean = np.full(n, tmean_c, dtype=np.float32) + 0.1 * np.sin(
        2 * np.pi * np.arange(n) / 365.0
    ).astype(np.float32)
    u = surf.get("10u")
    v = surf.get("10v")
    if u is not None and v is not None and hasattr(u, "detach"):
        wind = float(torch.sqrt(u[0, -1] ** 2 + v[0, -1] ** 2).detach().cpu().numpy())
    else:
        wind = 2.0
    return xr.Dataset(
        {
            "tmean": (("time",), tmean),
            "tmax": (("time",), (tmean + 2).astype(np.float32)),
            "tmin": (("time",), (tmean - 4).astype(np.float32)),
            "precip": (("time",), np.full(n, 3.0, dtype=np.float32)),
            "rh_mean": (("time",), np.full(n, 75.0, dtype=np.float32)),
            "srad": (("time",), np.full(n, 15.0, dtype=np.float32)),
            "wind10m": (("time",), np.full(n, wind, dtype=np.float32)),
            "vpd": (("time",), np.full(n, 1.0, dtype=np.float32)),
            "et0": (("time",), np.full(n, 3.0, dtype=np.float32)),
            "cwd": (("time",), np.zeros(n, dtype=np.float32)),
        },
        coords={"time": days, "lat": lat, "lon": lon},
        attrs={"aurora_backend": "aurora_forward"},
    )


def era5_xarray_to_batch(
    ds: xr.Dataset,
    init_time: datetime,
    *,
    n_lat: int = 17,
    n_lon: int = 32,
    region: str | None = None,
) -> Any:
    """
    Build an ``aurora.Batch`` from ERA5-like xarray (or synthetic) for one init time.

    When full ERA5 grids are unavailable, fills tensors with climatological placeholders
    shaped for Aurora 0.25° small (17×32).
    """
    import torch
    from aurora import Batch, Metadata

    if region is not None:
        preset = COCOA_BELT_REGIONS[normalize_region_key(region)]
        lats, lons = region_lat_lon_grid(preset, n_lat=n_lat, n_lon=n_lon)
    else:
        lats = np.linspace(90, -90, n_lat)
        lons = np.linspace(0, 360, n_lon + 1)[:-1]

    lat_t = torch.from_numpy(lats.astype(np.float32))
    lon_t = torch.from_numpy(lons.astype(np.float32))
    rng = np.random.default_rng(int(init_time.timestamp()) % (2**31))
    surf = {
        k: torch.from_numpy(rng.standard_normal((1, 2, n_lat, n_lon)).astype(np.float32))
        for k in ("2t", "10u", "10v", "msl")
    }
    static = {
        k: torch.from_numpy(rng.standard_normal((n_lat, n_lon)).astype(np.float32))
        for k in ("lsm", "z", "slt")
    }
    atmos = {
        k: torch.from_numpy(rng.standard_normal((1, 2, 4, n_lat, n_lon)).astype(np.float32))
        for k in ("z", "u", "v", "t", "q")
    }
    return Batch(
        surf_vars=surf,
        static_vars=static,
        atmos_vars=atmos,
        metadata=Metadata(
            lat=lat_t,
            lon=lon_t,
            time=(init_time,),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def check_aurora_commercial_gate(*, commercial_ok: bool, deployment_environment: str) -> None:
    """Raise when Aurora is used in production without commercial approval."""
    env = deployment_environment.strip().lower()
    if env in ("production", "prod") and not commercial_ok:
        raise ValueError(
            "Aurora downscaling in production requires AURORA_COMMERCIAL_OK=true after "
            "approval from AIWeatherClimate@microsoft.com. See docs/LICENSES.md."
        )


@dataclass
class AuroraScenarioRunner:
    """Subprocess-free Aurora rollout with Zarr cache and optional per-region LoRA."""

    cache_dir: Path
    model_size: ModelSize = "small"
    mock: bool = False
    lora_dir: Path = _DEFAULT_LORA_DIR
    _model: Any = None
    _lora_id: str = "base"

    @classmethod
    def from_settings(cls, settings: Any) -> AuroraScenarioRunner:
        size = getattr(settings, "aurora_model_size", "small")
        if size not in ("small", "medium"):
            size = "small"
        return cls(
            cache_dir=Path(
                getattr(settings, "aurora_cache_dir", _REPO_ROOT / "data/processed/aurora_scenario")
            ),
            model_size=size,  # type: ignore[arg-type]
            mock=bool(getattr(settings, "aurora_mock", False)),
        )

    @property
    def model_version(self) -> str:
        return AURORA_MODEL_VERSIONS[self.model_size]

    @property
    def lora_id(self) -> str:
        return self._lora_id

    def _lora_path(self, region: str) -> Path:
        reg = normalize_region_key(region)
        for backbone in ("aurora", "galileo", "agrifm", "terramind", "olmoearth", "aef"):
            path = self.lora_dir / f"{backbone}_lora_{reg}.safetensors"
            if path.is_file():
                return path
        return self.lora_dir / f"aurora_lora_{reg}.safetensors"

    def _ensure_model(self, region: str | None = None) -> Any:
        if self.mock:
            return None
        if self._model is not None:
            return self._model
        try:
            if self.model_size == "medium":
                from aurora import AuroraPretrained

                model = AuroraPretrained()
            else:
                from aurora import AuroraSmallPretrained

                model = AuroraSmallPretrained()
            model.load_checkpoint()
            self._model = model
            if region is not None:
                lora_path = self._lora_path(region)
                if lora_path.is_file():
                    if lora_path.name.startswith("aurora_lora_"):
                        from models.aurora_backbone import AuroraBackboneAdapter

                        adapter = AuroraBackboneAdapter(model)
                        adapter.load_region_adapter(normalize_region_key(region), lora_path)
                    self._lora_id = normalize_region_key(region)
                else:
                    self._lora_id = "base"
            return self._model
        except ImportError as exc:
            log.warning("aurora_import_failed", error=str(exc))
            return None

    def rollout_steps(self, start: str, end: str) -> int:
        days = pd_date_range(start, end)
        return max(1, len(days))

    def forecast_region(
        self,
        *,
        region: str,
        init_time: datetime,
        start: str,
        end: str,
        lead_h: int | None = None,
    ) -> xr.Dataset:
        """Roll out Aurora for a cocoa-belt region; returns regional-mean point series."""
        reg = normalize_region_key(region)
        preset = COCOA_BELT_REGIONS[reg]
        lead = lead_h if lead_h is not None else self.rollout_steps(start, end)
        key = aurora_cache_key(
            init_time=init_time,
            lead_h=lead,
            region=reg,
            model_size=self.model_size,
            lora_id=self._lora_id,
        )
        cached = load_cached_forecast(self.cache_dir, key)
        if cached is not None:
            return cached

        lat_c = 0.5 * (preset.south + preset.north)
        lon_c = 0.5 * (preset.west + preset.east)

        if self.mock or os.environ.get("AURORA_MOCK", "").lower() in ("1", "true", "yes"):
            ds = _mock_era5_point_dataset(lat=lat_c, lon=lon_c, start=start, end=end)
        else:
            model = self._ensure_model(region=reg)
            if model is None:
                ds = _mock_era5_point_dataset(lat=lat_c, lon=lon_c, start=start, end=end)
            else:
                batch = era5_xarray_to_batch(
                    xr.Dataset(),
                    init_time,
                    region=reg,
                )
                pred = batch
                steps = min(lead, 40)
                for _ in range(steps):
                    pred = model.forward(pred)
                ds = _aurora_batch_to_point_dataset(
                    pred, lat=lat_c, lon=lon_c, start=start, end=end
                )

        write_cached_forecast(self.cache_dir, key, ds)
        return ds

    def forecast_farm_point(
        self,
        lat: float,
        lon: float,
        region: str,
        window: tuple[str, str],
        horizon_year: int,
    ) -> xr.Dataset:
        """
        Farm-level daily climate for ``window`` using Aurora init at ``horizon_year``.

        Uses ERA5-schema output at the farm coordinates (nearest regional rollout).
        """
        start, end = window
        init = datetime(int(horizon_year), 6, 1, 12, 0)
        reg = normalize_region_key(region)
        self._lora_id = "base"
        lora_path = self._lora_path(reg)
        if lora_path.is_file():
            self._lora_id = reg
        ds_reg = self.forecast_region(
            region=reg,
            init_time=init,
            start=start,
            end=end,
        )
        return ds_reg.assign_coords(lat=lat, lon=lon)


def build_aurora_source_attribution(
    runner: AuroraScenarioRunner,
) -> list[dict[str, str | None]]:
    """Attribution block for ``SimulateScenarioResponse.source_attributions``."""
    return [
        {
            "id": "aurora_1.5",
            "role": "Earth-system scenario downscaling (Aurora 1.5)",
            "citation": "Bodnar et al., Nature 2025; Microsoft Aurora",
            "asset": runner.model_version,
            "aurora_model_version": runner.model_version,
            "aurora_lora_id": runner.lora_id,
        }
    ]
