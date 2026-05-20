"""
ICCO / ICE cocoa pricing, FX, and farm-gate pass-through for financial valuation.

Caches daily series under ``data/cache/finance/*.parquet``. Network fetches are
best-effort; stale cache or deterministic fallbacks keep CI and offline runs working.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / "data" / "cache" / "finance"

ICCO_CACHE = CACHE_DIR / "icco_daily.parquet"
FX_GHS_CACHE = CACHE_DIR / "fx_usd_ghs_daily.parquet"
FX_XOF_CACHE = CACHE_DIR / "fx_usd_xof_daily.parquet"
FUTURES_CACHE = CACHE_DIR / "ice_cocoa_futures.parquet"

# ICCO NY board indicative (USD/tonne) — updated when live ingest unavailable
DEFAULT_ICCO_NY_USD_PER_TONNE = 8_200.0
# ICE Cocoa (CC) spot proxy USD/tonne for forward curve anchoring
DEFAULT_ICE_SPOT_USD_PER_TONNE = 8_150.0

# Farm-gate pass-through vs ICCO NY (country-specific logistics / quality discounts)
COUNTRY_PASS_THROUGH: dict[str, float] = {
    "GHA": 0.72,
    "CIV": 0.65,
    "CMR": 0.81,
}

FORWARD_MONTHS: tuple[int, ...] = (3, 6, 12)
# Simple contango (% per month) when live futures curve unavailable
DEFAULT_MONTHLY_CONTANGO = 0.0025

PricingBasis = Literal["spot", "12m_forward", "trailing_3y_avg"]
SupportedCurrency = Literal["USD", "GHS", "XOF", "EUR"]

EXCHANGERATE_HOST = "https://api.exchangerate.host"
CACHE_MAX_AGE_DAYS = 1


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _cache_stale(path: Path, *, max_age_days: int = CACHE_MAX_AGE_DAYS) -> bool:
    if not path.is_file():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) > timedelta(days=max_age_days)


def _date_range_years(n_years: int = 5) -> pd.DatetimeIndex:
    end = date.today()
    start = end - timedelta(days=365 * n_years)
    return pd.date_range(start, end, freq="D")


def _synthetic_icco_series(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = len(index)
    noise = rng.normal(0, 80.0, n).cumsum() * 0.02
    level = DEFAULT_ICCO_NY_USD_PER_TONNE + noise
    return pd.DataFrame(
        {"date": index, "icco_ny_usd_per_tonne": np.clip(level, 2_000.0, 15_000.0)},
    )


def fetch_icco_daily(*, force: bool = False) -> pd.DataFrame:
    """
    Daily ICCO NY cocoa price (USD/tonne), cached in parquet.

    Live ingest uses World Bank commodity proxy when ICCO HTML is unavailable.
    """
    _ensure_cache_dir()
    if not force and not _cache_stale(ICCO_CACHE, max_age_days=7):
        return pd.read_parquet(ICCO_CACHE)

    index = _date_range_years(5)
    df = _synthetic_icco_series(index)

    try:
        # World Bank Pink Sheet–style open endpoint (commodity index proxy)
        url = (
            "https://api.worldbank.org/v2/country/all/indicator/PCOCOAUSD"
            "?format=json&per_page=2000&date=2018:2026"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list) and len(payload) > 1:
            rows = payload[1]
            if rows:
                wb = pd.DataFrame(rows)
                wb["date"] = pd.to_datetime(wb["date"], format="%Y")
                wb = wb.rename(columns={"value": "icco_ny_usd_per_tonne"})
                wb = wb.dropna(subset=["icco_ny_usd_per_tonne"])
                wb["icco_ny_usd_per_tonne"] = wb["icco_ny_usd_per_tonne"].astype(float)
                daily = wb.set_index("date").resample("D").ffill().reset_index()
                daily.columns = ["date", "icco_ny_usd_per_tonne"]
                if len(daily) >= 30:
                    df = daily
    except Exception as exc:
        logger.warning("ICCO live ingest failed (%s); using synthetic/cache fallback", exc)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df.to_parquet(ICCO_CACHE, index=False)
    return df


def _fetch_exchangerate_series(
    symbol: str,
    cache_path: Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """USD per 1 unit of ``symbol`` inverted to units of local currency per USD."""
    _ensure_cache_dir()
    if not force and not _cache_stale(cache_path, max_age_days=CACHE_MAX_AGE_DAYS):
        return pd.read_parquet(cache_path)

    rate = 1.0
    as_of = date.today()
    try:
        resp = requests.get(
            f"{EXCHANGERATE_HOST}/latest",
            params={"base": "USD", "symbols": symbol},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = float(data["rates"][symbol])
        as_of = datetime.strptime(data["date"], "%Y-%m-%d").date()
    except Exception as exc:
        logger.warning("FX fetch %s failed (%s); using fallback", symbol, exc)
        rate = {"GHS": 15.5, "XOF": 610.0, "EUR": 0.92}.get(symbol, 1.0)

    index = _date_range_years(3)
    df = pd.DataFrame(
        {
            "date": index,
            f"usd_per_{symbol.lower()}": rate,
        }
    )
    df.loc[df["date"].dt.date >= as_of, f"usd_per_{symbol.lower()}"] = rate
    df.to_parquet(cache_path, index=False)
    return df


def fetch_fx_rates(*, force: bool = False) -> dict[str, pd.DataFrame]:
    """Cached daily USD/GHS and USD/XOF (local units per 1 USD)."""
    ghs = _fetch_exchangerate_series("GHS", FX_GHS_CACHE, force=force)
    xof = _fetch_exchangerate_series("XOF", FX_XOF_CACHE, force=force)
    return {"GHS": ghs, "XOF": xof}


def _latest_rate(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return 1.0
    return float(df.sort_values("date").iloc[-1][col])


def fetch_forward_curve(
    *,
    as_of: date | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    ICE Cocoa futures–style forward curve (USD/tonne) for 3/6/12 months.

    Columns: ``as_of``, ``tenor_months``, ``price_usd_per_tonne``.
    """
    _ensure_cache_dir()
    as_of = as_of or date.today()
    if not force and not _cache_stale(FUTURES_CACHE, max_age_days=7):
        cached = pd.read_parquet(FUTURES_CACHE)
        if not cached.empty:
            return cached

    icco = fetch_icco_daily()
    spot = float(icco.sort_values("date").iloc[-1]["icco_ny_usd_per_tonne"])
    rows = []
    for months in FORWARD_MONTHS:
        premium = (1.0 + DEFAULT_MONTHLY_CONTANGO) ** months
        rows.append(
            {
                "as_of": pd.Timestamp(as_of),
                "tenor_months": months,
                "price_usd_per_tonne": spot * premium,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(FUTURES_CACHE, index=False)
    return df


def farm_gate_price_usd(
    icco_ny_usd: float,
    country_code: str,
) -> float:
    """ICCO NY × country pass-through factor."""
    factor = COUNTRY_PASS_THROUGH.get(country_code.upper(), COUNTRY_PASS_THROUGH["CIV"])
    return max(0.0, float(icco_ny_usd)) * factor


def trailing_3y_avg_icco(*, as_of: date | None = None) -> float:
    as_of = as_of or date.today()
    icco = fetch_icco_daily()
    icco = icco[icco["date"] <= pd.Timestamp(as_of)]
    window = icco[icco["date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=365 * 3)]
    if window.empty:
        return DEFAULT_ICCO_NY_USD_PER_TONNE
    return float(window["icco_ny_usd_per_tonne"].mean())


def resolve_price_usd_per_tonne(
    *,
    pricing_basis: PricingBasis = "spot",
    farm_gate: bool = True,
    country_code: str = "CIV",
    price_override_usd: float | None = None,
    as_of: date | None = None,
) -> float:
    """Resolve USD/tonne for avoided-loss valuation."""
    if price_override_usd is not None and price_override_usd >= 0:
        base = float(price_override_usd)
        if farm_gate and pricing_basis == "spot":
            # Override treated as ICCO NY when farm_gate=True
            return farm_gate_price_usd(base, country_code)
        return base

    as_of = as_of or date.today()
    if pricing_basis == "trailing_3y_avg":
        icco = trailing_3y_avg_icco(as_of=as_of)
    elif pricing_basis == "12m_forward":
        curve = fetch_forward_curve(as_of=as_of)
        row = curve.loc[curve["tenor_months"] == 12]
        icco = float(row.iloc[0]["price_usd_per_tonne"]) if len(row) else DEFAULT_ICE_SPOT_USD_PER_TONNE
    else:
        icco = float(
            fetch_icco_daily()
            .sort_values("date")
            .iloc[-1]["icco_ny_usd_per_tonne"]
        )

    return farm_gate_price_usd(icco, country_code) if farm_gate else icco


def price_per_tonne_usd(
    *,
    pricing_basis: PricingBasis = "spot",
    farm_gate: bool = True,
    country_code: str = "CIV",
) -> float:
    """Public alias for :func:`resolve_price_usd_per_tonne` without override."""
    return resolve_price_usd_per_tonne(
        pricing_basis=pricing_basis,
        farm_gate=farm_gate,
        country_code=country_code,
    )


def convert_usd_amount(
    amount_usd: float,
    currency: SupportedCurrency,
    *,
    fx: dict[str, pd.DataFrame] | None = None,
) -> float:
    """Convert a USD amount into ``currency`` using latest cached FX."""
    if currency == "USD":
        return amount_usd
    fx = fx or fetch_fx_rates()
    if currency == "GHS":
        col = "usd_per_ghs"
        rate = _latest_rate(fx["GHS"], col)
        return amount_usd * rate
    if currency == "XOF":
        col = "usd_per_xof"
        rate = _latest_rate(fx["XOF"], col)
        return amount_usd * rate
    if currency == "EUR":
        try:
            resp = requests.get(
                f"{EXCHANGERATE_HOST}/latest",
                params={"base": "USD", "symbols": "EUR"},
                timeout=10,
            )
            eur_per_usd = float(resp.json()["rates"]["EUR"])
        except Exception:
            eur_per_usd = 0.92
        return amount_usd * eur_per_usd
    raise ValueError(f"Unsupported currency: {currency}")


def infer_country_code(lat: float, lon: float) -> str:
    """Heuristic ISO3 cocoa producer from coordinates."""
    if -8.5 <= lon <= -2.5 and 4.0 <= lat <= 11.0:
        return "CIV"
    if -3.5 <= lon <= 2.0 and 4.0 <= lat <= 11.0:
        return "GHA"
    if 8.0 <= lon <= 16.5 and 1.5 <= lat <= 13.0:
        return "CMR"
    return "CIV"
