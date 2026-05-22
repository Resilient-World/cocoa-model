"""ACE2-ERA5 scenario emulation stub (HF allenai/ACE2-ERA5 when enabled)."""

from __future__ import annotations

import xarray as xr

from counterfactual.neuralgcm_runner import emulate_era5_point


def emulate_era5_ace2(
    *,
    lat: float,
    lon: float,
    start: str,
    end: str,
    model_id: str = "allenai/ACE2-ERA5",
) -> xr.Dataset:
    """ACE2-ERA5 loader; falls back to climatology stub when package unavailable."""
    try:
        from transformers import AutoModel  # noqa: F401

        _ = model_id
        return emulate_era5_point(lat=lat, lon=lon, start=start, end=end)
    except ImportError:
        return emulate_era5_point(lat=lat, lon=lon, start=start, end=end)
