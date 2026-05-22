# Typing playbook (mypy ratchet)

This repo uses a **ratchet** for gradual strict typing: each PR should **decrease** total type errors or move modules from `gradual_modules` into `strict_enabled` in [`pyproject.toml`](../pyproject.toml). Never reintroduce blanket `ignore_errors = true` on application packages.

## Commands

| Command | Purpose |
|---------|---------|
| `make typecheck` | `mypy src/` — global strict with per-module `disable_error_code` on `gradual_modules` |
| `make typecheck-strict` | `mypy --strict` only on `[tool.cocoa.typing].strict_enabled` |
| `pre-commit run --all-files` | Ruff + ratchet mypy + hygiene hooks |
| `python scripts/mypy_discover_gradual.py` | Regenerate `gradual_modules` after large refactors |
| `python scripts/check_no_prints.py` | CI guard: no bare `print()` under `src/` |

## Registry (`[tool.cocoa.typing]`)

```toml
[tool.cocoa.typing]
strict_enabled = [ ... ]   # must pass: make typecheck-strict
gradual_modules = [ ... ]  # ranked outer → inner; paired with [[tool.mypy.overrides]]
```

**Strict-enabled (Sprint 1):** `api.config`, `api.schemas`, `analysis._report`, `models.conformal.cqr` (public alias: `models.cqr` shim).

## Gradual overrides

Modules in `gradual_modules` share:

```toml
disable_error_code = ["misc", "arg-type", "return-value", ...]
```

Phase 1 also suppresses common legacy codes (`attr-defined`, `import-untyped`, etc.) documented in `pyproject.toml`. Remove codes from this list as you fix modules — do not add new blanket `ignore_errors`.

## Ratchet rules for PRs

1. **Strict modules:** `mypy --strict -p <module>` must stay at **0 errors**. CI enforces via `make typecheck-strict`.
2. **No regressions:** Do not increase error counts on modules already in `strict_enabled`.
3. **Promotion:** To strict-enable a module:
   - Fix all errors with `mypy --strict -p module`
   - Add module to `strict_enabled`
   - Remove from `gradual_modules` and from the `[[tool.mypy.overrides]]` `module = [...]` list
4. **Discovery:** After refactors touching types, run:
   ```bash
   MYPYPATH=src mypy --strict src/ 2>&1 | tee /tmp/mypy.txt
   python scripts/mypy_discover_gradual.py /tmp/mypy.txt
   ```

## Cocoa-specific patterns

### xarray `DataArray` with named dims

Prefer documenting dims in docstrings (`time`, `lat`, `lon`) and narrowing at API boundaries:

```python
from typing import cast
import xarray as xr

def as_daily_climate(ds: xr.Dataset) -> xr.DataArray:
    da = ds["tmax_c"]
    assert tuple(da.dims) == ("time",)
    return da
```

### Pandera `DataFrame[Schema]`

Farm panels and ingest manifests use schemas in [`src/data/schemas.py`](../src/data/schemas.py):

```python
from pandera.typing import DataFrame
from data.schemas import FarmPanelSchema

def load_panel(path: Path) -> DataFrame[FarmPanelSchema]:
    ...
```

### Earth Engine (`ee`)

Keep `ignore_missing_imports` for `ee.*` in `pyproject.toml`. In app code, use `TYPE_CHECKING` protocols or `ee: Any` at boundaries — do not import `ee` in strict modules unless required.

### PyTorch / surrogates

Keep heavy torch submodules on `gradual_modules` until promoted. Strict service layers (`api.config`, `api.schemas`) should not pull full surrogate typing into the same PR.

## Per-module `# type: ignore`

Use only when genuinely impossible (C extension, dynamic plugin). Prefer:

```python
value = external()  # type: ignore[no-untyped-call]  # TICKET-123
```

Never use bare `# type: ignore` (warned by `warn_unused_ignores`).

## Ruff ratchet

`[tool.ruff.lint]` uses `E,F,B,I,UP,N,ANN,SIM,RUF` with phase-1 ignores for legacy ANN/N/RUF noise. Tighten ignores in follow-up PRs; tests allow `ANN` and `S101` via `per-file-ignores`.
