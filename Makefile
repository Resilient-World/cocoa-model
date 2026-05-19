.PHONY: attrici-env attrici-counterfactual

attrici-env:
	python -m venv .venv-attrici
	.venv-attrici/bin/pip install --upgrade pip
	.venv-attrici/bin/pip install "attrici @ git+https://github.com/ISI-MIP/attrici@v2.0.1"
	.venv-attrici/bin/pip install scipy numpy xarray netCDF4 pandas

# Custom counterfactual (ERA5-Land / post-2019) — requires VAR and SSA-smoothed GMT
attrici-counterfactual:
	python scripts/run_attrici_subprocess.py \
		--factual data/era5_factual/$(VAR).nc \
		--out data/counterfactual/$(VAR).nc \
		--variable $(VAR) --backend scipy
