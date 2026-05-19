.PHONY: attrici-env

attrici-env:
	python -m venv .venv-attrici
	.venv-attrici/bin/pip install --upgrade pip
	.venv-attrici/bin/pip install "attrici @ git+https://github.com/ISI-MIP/attrici@v2.0.1"
	.venv-attrici/bin/pip install scipy numpy xarray netCDF4
