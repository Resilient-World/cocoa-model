# Parametric insurance product

`POST /price-parametric` prices a farm-level yield trigger for cocoa resilience contracts.

Example request:

```json
{
  "farm_location": {"lat": 6.5, "lon": -1.2},
  "farm_size_ha": 5.0,
  "strike_t_per_ha": 1.2,
  "coverage_horizon_years": 1,
  "scenario": "baseline",
  "cocoa_price_usd": 3000
}
```

The endpoint returns:

- `fair_premium_usd`: Monte Carlo expected payout.
- `loaded_premium_usd`: expected payout plus DVDS Λ=1.5 tail-risk loading and conformal volatility.
- `basis_risk_r2`: fit between realized loss tonnes and parametric payout tonnes.
- `lambda_sensitivity`: premium sensitivity at selected marginal-sensitivity bounds.

Pricing math lives in `finance.parametric_insurance`:

- `compute_basis_risk(realized_loss_t, parametric_payout_t)` returns correlation, RMSE, regression slope, and R².
- `price_parametric_trigger(...)` computes strike shortfall payouts and DVDS/conformal loading.
- `smile_corrected_pricing(...)` applies out-of-the-money volatility smile multipliers for drought-only or harmattan-only triggers.

Underwriting note: low basis-risk R² means the trigger needs redesign or more local index calibration before customer quoting.
