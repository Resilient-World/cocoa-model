# GEDI/ICESat-2 canopy features

## Products

- **GEDI L4A biomass** (`LARSE/GEDI/GEDI04_A_002_MONTHLY`): aboveground biomass density, reported as `agb_mg_ha`.
- **GEDI L3 canopy height** (`LARSE/GEDI/GEDI03_001/GEDI03_canopy_height`): 1 km canopy height, reported as `canopy_height_m`.
- **ICESat-2 ATL08**: canopy height / cover point support through an Earthdata Subsetter subprocess configured by `ATL08_SUBSETTER_CMD`.

`src/data/gedi_canopy.py` exposes `GEDICanopyIngest(aoi, year).build()` and `sample_canopy_at_point(lat, lon, year)`. The API response includes `canopy_height_m`, `canopy_cover_pct`, `agb_mg_ha`, `height_uncertainty_m`, `gedi_n_shots`, and `source_attributions`.

## Model wiring

The feature resolver adds two normalized static site features:

- `canopy_height_norm = canopy_height_m / 45`
- `agb_norm = agb_mg_ha / 500`

The yield surrogate site vector is now 15 fields: the prior 13-field layout plus canopy height and AGB before tree-age cohort fields.

## Caveats

GEDI and ATL08 are sparse-footprint LiDAR products. Cocoa belts with frequent cloud cover, steep topography, or smallholder mosaics may have few shots near a farm point. Use `gedi_n_shots` and `height_uncertainty_m` as quality indicators, and prefer aggregated AOI features over single-shot interpretations where possible.
