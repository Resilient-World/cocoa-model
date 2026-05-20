"""
ATTRICI counterfactual climate runner (subprocess boundary).

ATTRICI is licensed under GPLv3. This module **never** imports ``attrici`` or any
``attrici.*`` submodule so the Resilient Cocoa codebase can remain MIT-licensed.
All ATTRICI work is delegated to the ``attrici`` CLI via :class:`subprocess.run`.

Install ATTRICI only in an isolated environment (e.g. ``pip install -e '.[attrici]'``)
for integration tests; it is not a required runtime dependency of this package.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

# ERA5 / ingest names requested for counterfactual detrending
SUPPORTED_VARIABLES: frozenset[str] = frozenset(
    {"tmax", "tmin", "precip", "rh_mean", "srad", "wind10m"}
)

# ATTRICI v2 ISIMIP short names (subprocess ``--variable`` argument)
_ERA5_TO_ATTRICI: dict[str, str] = {
    "tmax": "tasmax",
    "tmin": "tasmin",
    "precip": "pr",
    "rh_mean": "hurs",
    "srad": "rsds",
    "wind10m": "sfcwind",
}


def _open_xarray():
    import xarray as xr

    return xr


class CounterfactualClimateProvider(Protocol):
    """Interface for point-wise counterfactual climate access."""

    def get(self, lat: float, lon: float, year: int) -> xr.Dataset:
        """Return counterfactual daily (or annual) climate for one site and year."""
        ...


class ATTRICIRunner:
    """
    Run ATTRICI detrend on factual Zarr slices and persist counterfactuals to Zarr.

    Parameters
    ----------
    attrici_bin:
        ATTRICI CLI executable (default ``"attrici"`` on ``PATH``).
    gmt_file:
        SSA-smoothed global-mean temperature NetCDF for ATTRICI.
    work_dir:
        Temporary NetCDF, logs, and intermediate outputs.
    n_workers:
        Passed to ATTRICI ``--workers``.
    """

    def __init__(
        self,
        gmt_file: Path,
        work_dir: Path,
        attrici_bin: str = "attrici",
        n_workers: int = 4,
    ) -> None:
        self.attrici_bin = attrici_bin
        self.gmt_file = Path(gmt_file)
        self.work_dir = Path(work_dir)
        self.n_workers = n_workers
        self._logs_dir = self.work_dir / "logs"

    def _attrici_version(self) -> str:
        result = subprocess.run(
            [self.attrici_bin, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "attrici --version failed (code %s): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return "unknown"
        return (result.stdout or result.stderr).strip()

    def _run_attrici_cli(
        self,
        *,
        era5_var: str,
        tmp_nc: Path,
        tmp_out_nc: Path,
    ) -> None:
        attrici_var = _ERA5_TO_ATTRICI[era5_var]
        log_path = self._logs_dir / f"{era5_var}.log"
        cmd = [
            self.attrici_bin,
            "--gmt",
            str(self.gmt_file),
            "--input",
            str(tmp_nc),
            "--variable",
            attrici_var,
            "--output",
            str(tmp_out_nc),
            "--workers",
            str(self.n_workers),
        ]
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"# command: {' '.join(cmd)}\n\n")
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                None,
                f"ATTRICI failed for {era5_var}; see {log_path}",
            )

    def _materialize_variable_nc(
        self,
        factual: xr.Dataset,
        era5_var: str,
        tmp_nc: Path,
    ) -> None:
        if era5_var not in factual.data_vars:
            raise KeyError(
                f"Variable {era5_var!r} not in factual Zarr "
                f"(available: {list(factual.data_vars)})"
            )
        slice_ds = factual[[era5_var]]
        slice_ds.to_netcdf(tmp_nc)

    def _read_counterfactual_nc(self, tmp_out_nc: Path, era5_var: str) -> xr.Dataset:
        xr = _open_xarray()
        out = xr.open_dataset(tmp_out_nc)
        if era5_var in out.data_vars:
            return out[[era5_var]]
        # ATTRICI may emit ISIMIP / internal names (e.g. cfact, tasmax)
        attrici_var = _ERA5_TO_ATTRICI[era5_var]
        for candidate in (era5_var, attrici_var, "cfact"):
            if candidate in out.data_vars:
                renamed = out[[candidate]].rename({candidate: era5_var})
                return renamed
        raise KeyError(
            f"No counterfactual variable found in {tmp_out_nc} "
            f"(tried {era5_var}, {attrici_var}, cfact); got {list(out.data_vars)}"
        )

    def run(
        self,
        factual_zarr: Path,
        variables: Sequence[str],
        output_zarr: Path,
        overwrite: bool = False,
    ) -> Path:
        """
        Detrend each requested variable and write groups into ``output_zarr``.

        Parameters
        ----------
        factual_zarr:
            Path to factual daily climate Zarr (ERA5-Land style variable names).
        variables:
            Subset of :data:`SUPPORTED_VARIABLES`.
        output_zarr:
            Consolidated Zarr store (one group per variable).
        overwrite:
            Replace an existing ``output_zarr`` when True.

        Returns
        -------
        pathlib.Path
            ``output_zarr`` after all variables are written.
        """
        xr = _open_xarray()
        factual_zarr = Path(factual_zarr)
        output_zarr = Path(output_zarr)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        requested = [v for v in variables if v in SUPPORTED_VARIABLES]
        skipped = set(variables) - set(requested)
        if skipped:
            logger.warning("Ignoring unsupported counterfactual variables: %s", sorted(skipped))
        if not requested:
            raise ValueError(
                f"No supported variables in {list(variables)}; "
                f"expected subset of {sorted(SUPPORTED_VARIABLES)}"
            )

        if output_zarr.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output Zarr already exists: {output_zarr} (pass overwrite=True)"
                )
            import shutil

            if output_zarr.is_dir():
                shutil.rmtree(output_zarr)
            else:
                output_zarr.unlink()

        attrici_version = self._attrici_version()
        factual = xr.open_zarr(factual_zarr)

        tmp_dir = self.work_dir / "nc_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for idx, era5_var in enumerate(requested):
            tmp_nc = tmp_dir / f"factual_{era5_var}.nc"
            tmp_out_nc = tmp_dir / f"counterfactual_{era5_var}.nc"

            logger.info("ATTRICI detrend %s (%s)", era5_var, _ERA5_TO_ATTRICI[era5_var])
            self._materialize_variable_nc(factual, era5_var, tmp_nc)
            self._run_attrici_cli(era5_var=era5_var, tmp_nc=tmp_nc, tmp_out_nc=tmp_out_nc)

            cf_ds = self._read_counterfactual_nc(tmp_out_nc, era5_var)
            cf_ds.attrs["counterfactual"] = True
            cf_ds.attrs["attrici_version"] = attrici_version
            cf_ds.attrs["source_variable"] = era5_var

            mode = "w" if idx == 0 else "a"
            cf_ds.to_zarr(output_zarr, group=era5_var, mode=mode)

        # Root-level metadata for :func:`load_counterfactual`
        root_attrs = {
            "counterfactual": True,
            "attrici_version": attrici_version,
            "variables": requested,
        }
        try:
            import zarr

            root = zarr.open_group(output_zarr, mode="a")
            root.attrs.update(root_attrs)
        except Exception as exc:
            logger.warning("Could not write root Zarr attrs: %s", exc)

        return output_zarr


def load_counterfactual(zarr_path: Path | str) -> xr.Dataset:
    """
    Open a consolidated counterfactual Zarr produced by :class:`ATTRICIRunner`.

    Merges all variable groups into a single :class:`xarray.Dataset`.
    """
    xr = _open_xarray()
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Counterfactual Zarr not found: {zarr_path}")

    root_attrs: dict[str, object] = {}
    group_names: list[str] = []
    try:
        import zarr

        root = zarr.open_group(zarr_path, mode="r")
        group_names = sorted(root.group_keys())
        root_attrs = dict(root.attrs)
    except Exception:
        pass

    if group_names:
        pieces = [xr.open_zarr(zarr_path, group=name) for name in group_names]
        merged = xr.merge(pieces, compat="override")
    else:
        merged = xr.open_zarr(zarr_path)

    merged.attrs.setdefault("counterfactual", True)
    if "attrici_version" in root_attrs:
        merged.attrs.setdefault("attrici_version", root_attrs["attrici_version"])
    return merged


class ZarrCounterfactualProvider:
    """:class:`CounterfactualClimateProvider` backed by a consolidated Zarr store."""

    def __init__(self, zarr_path: Path | str) -> None:
        self.zarr_path = Path(zarr_path)

    def get(self, lat: float, lon: float, year: int) -> xr.Dataset:
        ds = load_counterfactual(self.zarr_path)
        lat_name = "latitude" if "latitude" in ds.dims or "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.dims or "longitude" in ds.coords else "lon"
        if lat_name not in ds.dims and lat_name not in ds.coords:
            raise ValueError(f"No latitude coordinate in counterfactual store: {self.zarr_path}")
        if lon_name not in ds.dims and lon_name not in ds.coords:
            raise ValueError(f"No longitude coordinate in counterfactual store: {self.zarr_path}")

        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        if "time" in point.dims or "time" in point.coords:
            point = point.sel(time=point["time"].dt.year == year)
        return point


def main(argv: list[str] | None = None) -> int:
    """CLI: ATTRICI counterfactual detrend for factual ERA5 Zarr."""
    import argparse
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ATTRICI counterfactual climate runner")
    parser.add_argument("--input", type=Path, required=True, help="Factual ERA5 Zarr")
    parser.add_argument("--gmt", type=Path, required=True, help="GMT SSA NetCDF for ATTRICI")
    parser.add_argument("--out", type=Path, required=True, help="Output counterfactual Zarr")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(SUPPORTED_VARIABLES),
        help="ERA5 variables to detrend",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    work_dir = Path(tempfile.mkdtemp(prefix="attrici_runner_"))
    runner = ATTRICIRunner(gmt_file=args.gmt, work_dir=work_dir)
    try:
        runner.run(
            args.input,
            args.variables,
            args.out,
            overwrite=args.overwrite,
        )
        logger.info("Wrote counterfactual Zarr to %s", args.out)
        return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
