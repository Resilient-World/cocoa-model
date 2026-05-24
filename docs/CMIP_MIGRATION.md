# CMIP6 to CMIP7 migration

CMIP scenario building now goes through `counterfactual.cmip_factory.build_cmip_scenario_builder`.

Default behavior is unchanged:

```bash
CMIP_VERSION=cmip6
```

When AR7/CMIP7 ensemble Zarrs are published and harmonized, switch with one environment variable:

```bash
export CMIP_VERSION=cmip7
export CMIP7_ZARR_PATH=data/processed/cmip7_ensemble.zarr
```

The CMIP7 placeholder recognizes AR7-style SSP labels:

- `SSP1-1.9`
- `SSP1-2.6`
- `SSP2-4.5`
- `SSP3-7.0`
- `SSP5-8.5`

Supported horizon conventions are 2030, 2050, 2080, and 2100. Until the configured CMIP7 Zarr exists, `CMIP_VERSION=cmip7` returns a clear "CMIP7 ensemble not yet published" warning/error rather than crashing deep in xarray.

Run `scripts/ingest_cmip7_when_available.py` to see the TODO checklist stub for future ingestion.
