"""
GEE-free ingestion of ENSO, Atlantic Niño (Atl3), and IOD monthly indices.

Caches a unified parquet at ``data/external/teleconnection_indices.parquet`` for
DVC and offline API resolution.

References
----------
- Klein, Lemordant, et al. (2023), Climatic Change — ENSO–Atlantic teleconnections
- NOAA CPC Niño 3.4 (ERSSTv5-based sstoi.indices)
- Columbia IRIDL HadISST Atl3
- NOAA/BOM Dipole Mode Index (IOD)
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from data.cocoa_exposure import REGIONS, normalize_region_key

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = _REPO_ROOT / "data" / "external" / "teleconnection_indices.parquet"

NINO34_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
# PSL Atlantic Niño / Atl3 (HadISST-based monthly index)
ATL3_URLS = (
    "https://psl.noaa.gov/psd/data/correlation/atl3.data",
    "https://www.esrl.noaa.gov/psd/data/correlation/atl3.data",
)
IOD_URLS = (
    "https://psl.noaa.gov/psd/data/correlation/iod.data",
    "https://www.esrl.noaa.gov/psd/data/correlation/iod.data",
)

CACHE_MAX_AGE_S = 7 * 24 * 3600

INDEX_NAMES = ("nino34", "atl3", "iod")

# Growing-year month lists: (calendar_year, month) tuples length 12
# West Africa bimodal: Oct(year-1) … Sep(year)
_WEST_AFRICA_REGIONS = frozenset({"ghana", "civ", "cameroon", "nigeria"})


def _growing_year_months(region: str, year: int) -> list[tuple[int, int]]:
    """Return 12 (year, month) pairs for the region's cocoa growing year ending in ``year``."""
    key = normalize_region_key(region)
    if key in _WEST_AFRICA_REGIONS:
        months: list[tuple[int, int]] = []
        for m in (10, 11, 12):
            months.append((year - 1, m))
        for m in range(1, 10):
            months.append((year, m))
        return months
    if key in ("ecuador", "peru", "colombia"):
        return [(year, m) for m in range(1, 13)]
    if key == "indonesia":
        return [(year - 1 if m >= 10 else year, m) for m in list(range(10, 13)) + list(range(1, 10))]
    return [(year, m) for m in range(1, 13)]


def _fetch_text(url: str, *, timeout: int = 90) -> str:
    headers = {"User-Agent": "resilient-cocoa-model/0.2 (research)"}
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    text = response.text.strip()
    if len(text) < 20:
        raise ValueError(f"Empty response from {url}")
    return text


def _fetch_first(urls: tuple[str, ...]) -> str:
    last_exc: Exception | None = None
    for url in urls:
        try:
            return _fetch_text(url)
        except Exception as exc:
            last_exc = exc
            logger.debug("Fetch failed for %s: %s", url, exc)
    raise RuntimeError(f"All URLs failed: {urls}") from last_exc


def _proxy_atl3_iod_from_nino34(nino34_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Fallback when Atl3/DMI downloads fail (Klein et al. 2023 co-variability structure).

    Atlantic Niño lags Pacific ENSO; IOD is partially independent but correlated at long leads.
    """
    s = nino34_df.set_index("time")["nino34"].sort_index()
    atl3 = (0.55 * s.shift(2) + 0.25 * s.shift(4)).interpolate(limit_direction="both")
    iod = (0.35 * s.shift(3) - 0.15 * s.shift(6)).interpolate(limit_direction="both")
    return atl3, iod


def parse_nino34_sstoi(text: str) -> pd.DataFrame:
    """Parse NOAA CPC ``sstoi.indices``; return anomaly column ``nino34`` (°C)."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "YR" in line.upper() or "NINO" in line.upper():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            y = int(parts[0])
            m = int(parts[1])
        except ValueError:
            continue
        try:
            # Columns: YR MON NINO1+2 NINO3 NINO4 NINO3.4 (anomaly last)
            val = float(parts[5])
        except ValueError:
            continue
        rows.append({"time": pd.Timestamp(year=y, month=m, day=1), "nino34": val})
    if not rows:
        raise ValueError("No Niño 3.4 rows parsed from sstoi.indices")
    df = pd.DataFrame(rows).sort_values("time").drop_duplicates("time")
    return df


def parse_atl3_psl(text: str) -> pd.DataFrame:
    """Parse NOAA PSL ``atl3.data`` monthly Atlantic Niño index."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            y = int(parts[0])
            m = int(parts[1])
            val = float(parts[2])
        except ValueError:
            continue
        rows.append({"time": pd.Timestamp(year=y, month=m, day=1), "atl3": val})
    if not rows:
        raise ValueError("No Atl3 rows parsed")
    return pd.DataFrame(rows).sort_values("time").drop_duplicates("time")


def parse_iod_bom(text: str) -> pd.DataFrame:
    """Parse BOM DMI text file (year, month, dipole mode index)."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "DMI" in line.upper():
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            continue
        try:
            y = int(parts[0])
            m = int(parts[1])
            val = float(parts[2])
        except ValueError:
            continue
        rows.append({"time": pd.Timestamp(year=y, month=m, day=1), "iod": val})
    if not rows:
        raise ValueError("No IOD rows parsed from DMI file")
    return pd.DataFrame(rows).sort_values("time").drop_duplicates("time")


def fetch_atl3_monthly() -> pd.DataFrame:
    """Download Atl3 monthly index (PSL correlation format)."""
    return parse_atl3_psl(_fetch_first(ATL3_URLS))


def fetch_iod_monthly() -> pd.DataFrame:
    """Download IOD / DMI monthly index (PSL correlation format)."""
    return parse_iod_bom(_fetch_first(IOD_URLS))


def build_indices_table(
    *,
    nino34_df: pd.DataFrame | None = None,
    atl3_df: pd.DataFrame | None = None,
    iod_df: pd.DataFrame | None = None,
    allow_proxy: bool = True,
) -> pd.DataFrame:
    """Merge index series on ``time``."""
    nino34_df = nino34_df if nino34_df is not None else parse_nino34_sstoi(_fetch_text(NINO34_URL))

    if atl3_df is None:
        try:
            atl3_df = fetch_atl3_monthly()
        except Exception as exc:
            if not allow_proxy:
                raise
            logger.warning("Atl3 ingest failed (%s); using ENSO-regressed proxy", exc)
            atl3_s, _ = _proxy_atl3_iod_from_nino34(nino34_df)
            atl3_df = pd.DataFrame({"time": atl3_s.index, "atl3": atl3_s.astype(np.float32).values})

    if iod_df is None:
        try:
            iod_df = fetch_iod_monthly()
        except Exception as exc:
            if not allow_proxy:
                raise
            logger.warning("IOD ingest failed (%s); using ENSO-regressed proxy", exc)
            _, iod_s = _proxy_atl3_iod_from_nino34(nino34_df)
            iod_df = pd.DataFrame({"time": iod_s.index, "iod": iod_s.astype(np.float32).values})

    out = nino34_df.merge(atl3_df, on="time", how="outer")
    out = out.merge(iod_df, on="time", how="outer")
    out = out.sort_values("time").reset_index(drop=True)
    for col in INDEX_NAMES:
        out[col] = out[col].astype(np.float32)
        out[col] = out[col].interpolate(method="linear", limit_direction="both")
    return out


def refresh_indices(
    path: Path | str | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download/merge indices and write parquet cache."""
    out_path = Path(path) if path is not None else DEFAULT_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not force:
        age = time.time() - out_path.stat().st_mtime
        if age < CACHE_MAX_AGE_S:
            logger.info("Teleconnection cache fresh (%.1f h); skipping download", age / 3600)
            return out_path

    logger.info("Fetching teleconnection indices (NOAA Niño3.4, PSL Atl3, BOM IOD)")
    table = build_indices_table()
    table.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d months)", out_path, len(table))
    return out_path


def load_indices_table(path: Path | str | None = None) -> pd.DataFrame:
    """Load cached parquet; refresh if missing."""
    out_path = Path(path) if path is not None else DEFAULT_PARQUET
    if not out_path.is_file():
        refresh_indices(out_path)
    df = pd.read_parquet(out_path)
    if "time" not in df.columns:
        raise ValueError(f"{out_path} missing 'time' column")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def get_indices_for_year(
    year: int,
    region: str,
    *,
    table: pd.DataFrame | None = None,
    path: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """
    Monthly index values for the region growing year ending in ``year``.

    Returns ``{nino34, atl3, iod}`` each ``np.ndarray`` shape ``[12]`` (float32).
    """
    df = table if table is not None else load_indices_table(path)
    months = _growing_year_months(region, year)
    series: dict[str, list[float]] = {k: [] for k in INDEX_NAMES}
    for cy, cm in months:
        ts = pd.Timestamp(year=cy, month=cm, day=1)
        row = df.loc[df["time"] == ts]
        if row.empty:
            row = df.iloc[(df["time"] - ts).abs().argsort()[:1]]
        for name in INDEX_NAMES:
            val = float(row[name].iloc[0])
            series[name].append(val)
    return {k: np.asarray(v, dtype=np.float32) for k, v in series.items()}


def region_key_from_latlon(lat: float, lon: float) -> str:
    """Pick :data:`REGIONS` key containing the farm."""
    for key, preset in REGIONS.items():
        if preset.west <= lon <= preset.east and preset.south <= lat <= preset.north:
            return key
    if -12.0 <= lon <= 5.0 and -12.0 <= lat <= 12.0:
        return "ghana" if lat >= 6.0 else "civ"
    return "ghana"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh teleconnection index parquet cache")
    parser.add_argument("--refresh", action="store_true", help="Force re-download")
    parser.add_argument("--path", type=Path, default=DEFAULT_PARQUET)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    refresh_indices(args.path, force=args.refresh)


if __name__ == "__main__":
    main()
