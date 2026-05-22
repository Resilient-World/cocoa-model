# Third-party data and software licenses

## ISIMIP3a counterclim dataset

**Source:** Mengel, M., Treu, S., Lange, S., Frieler, K. (2021). “ATTRICI v1.1 – counterfactual
climate for impact attribution”, *Geosci. Model Dev.* **14**, 5269–5284.
DOI: [10.5194/gmd-14-5269-2021](https://doi.org/10.5194/gmd-14-5269-2021)

Counterfactual GSWP3-W5E5 daily fields (`climate_scenario=counterclim`, ISIMIP3a) are
distributed via the [ISIMIP repository](https://data.isimip.org/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (ISIMIP terms).

**Fallback archive:** Zenodo record [5036364](https://zenodo.org/record/5036364) (Mengel et al.
2021b dataset).

**Ingest module:** `src/data/attrici_counterfactual.py` (no `attrici` Python import).

## Galileo (NASA Harvest / Mila / Ai2)

See [NOTICE.md](../NOTICE.md) at the repository root.

## ATTRICI (GPLv3 subprocess boundary)

See [docs/licensing/ATTRICI_GPL_BOUNDARY.md](licensing/ATTRICI_GPL_BOUNDARY.md).

## Microsoft Aurora 1.5 (Bodnar et al., Nature 2025)

**Source:** Microsoft Aurora earth-system foundation model
([microsoft/aurora](https://github.com/microsoft/aurora), PyPI `microsoft-aurora`).

**Research use:** Aurora weights and the `microsoft-aurora` package are licensed for
**research use** by default. See the upstream repository license and model card.

**Commercial use:** Contact **AIWeatherClimate@microsoft.com** before any commercial or
production deployment that runs Aurora inference.

**Runtime gate:** Set `AURORA_COMMERCIAL_OK=true` only after Microsoft has approved
commercial use. Production-like deployments (`OTEL_DEPLOYMENT_ENVIRONMENT=production` or
`prod`) reject `downscaling_method=aurora` when this flag is unset.

**Optional install:** `pip install -e ".[aurora]"` (does not install by default).
