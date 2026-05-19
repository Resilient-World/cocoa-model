# GSWP3-W5E5 factual climate (ISIMIP3a)

GSWP3-W5E5 v1.0 factual data (1901–2019, 0.5°, daily) is available from the ISIMIP repository:

<https://data.isimip.org/search/tree/ISIMIP3a/InputData/climate/atmosphere/gswp3-w5e5/>

## Required variables

`tas`, `tasmin`, `tasmax`, `pr`, `hurs`, `rsds`, `sfcwind`, `ps`

## Download

Install the [ISIMIP client](https://github.com/ISI-MIP/isimip-client) and run:

```bash
isimip-client download \
  --simulation_round ISIMIP3a \
  --product InputData \
  --climate_forcing gswp3-w5e5 \
  --climate_variable tas,tasmin,tasmax,pr,hurs,rsds,sfcwind,ps \
  --time_step daily \
  --target data/raw/gswp3-w5e5/
```

## License and citation

- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (ISIMIP terms).
- **Citation:** Lange et al. 2021 (W5E5 v2.0), Kim 2017 (GSWP3 v1.09), Mengel et al. 2021 (homogenization).

## GMT for ATTRICI

SSA-smoothed global mean temperature for detrending:

```bash
make attrici-env
.venv-attrici/bin/attrici ssa data/raw/gmt/gmt_raw.nc data/raw/gmt/ssa_gmt.nc \
  --variable tas --window-size 3650 --subset 10
```

(`--window-size` ≈ `ssa_window_years × 365` from `ATTRICIConfig`.)

## Counterfactual pipeline

```bash
pip install -e .
python scripts/run_attrici.py
```

Outputs: `data/counterfactual/` (0.5° ATTRICI). Downscale to the ERA5-Land grid with
`counterfactual.delta_downscaler.DeltaDownscaler`.
