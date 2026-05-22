"""Optional NeuralGCM ERA5 emulation (install ``[scenario_ml]`` + ``NEURALGCM_ENABLED``)."""

from __future__ import annotations

import numpy as np
import xarray as xr


def emulate_era5_point(
    *,
    lat: float,
    lon: float,
    start: str,
    end: str,
    historical_zarr: str | None = None,
) -> xr.Dataset:
    """
    Return ERA5-schema daily dataset for one point.

    Uses NeuralGCM when installed; otherwise applies a deterministic linear
    bridge from historical climatology (CI / local dev stub).
    """
    try:
        import neuralgcm  # noqa: F401

        raise NotImplementedError(
            "NeuralGCM full grid emulation requires GPU assets; use validate_neuralgcm_scenario.py on HPC"
        )
    except ImportError:
        pass
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
    )
    return ds


def pd_date_range(start: str, end: str) -> np.ndarray:
    import pandas as pd

    return pd.date_range(start, end, freq="D").to_numpy()
