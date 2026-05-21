# Cooperative policy targeting

This project supports two complementary ways to decide **which farms receive an intervention**:

1. **Greedy ranking** (`POST /rank-interventions`) — sort farms by estimated CATE (and optional cost) and spend a budget top-down.
2. **Interpretable rules** (`POST /learn-policy-rules`) — honest **doubly-robust policy trees/forests** (EconML; Athey & Wager 2021) that output regulator-readable if-then rules.

EconML is **MIT-licensed** and compatible with this repository’s MIT license.

## When to use which

| Approach | Best for | Output |
|----------|----------|--------|
| Greedy rank | Fixed budget, simple rollout lists | Ordered farm list |
| Policy tree | Board/regulator briefings, field manuals | IF–THEN rules + leaf statistics |
| Policy forest | Stability across bootstrap samples (offline) | Ensemble policy + rules from tree 0 |

Default API and laptop workflows should use **`learner=tree`**. Forests (`n_estimators=500`) are for offline reports via [`scripts/learn_targeting_rules.py`](../scripts/learn_targeting_rules.py).

## Reading a rule

Example:

```text
IF baseline_yield <= 1.2 AND slope_degrees > 8.0 THEN treat_with_shade_trees ELSE do_not_treat
```

- Conditions use **original column names** from your panel (never internal feature indices).
- `treat_with_shade_trees` is the `recommended_treatment_label` you pass to the API.
- Each leaf row in the rulebook includes `n_units`, `treat_fraction`, `expected_uplift`, and a **95% bootstrap CI** for that leaf.

## Honesty

`honest=True` is always used: half the sample builds the tree structure, half estimates leaf welfare (Athey & Wager 2021). This supports valid leaf-level uncertainty for cooperative reporting.

## Cost-aware learning

If you pass `cost_col`, the second policy stage maximizes **uplift minus cost** instead of raw CATE. The greedy baseline (when `budget` is set) still uses **CATE / cost** ranking. Document both numbers when presenting to managers: gross targeting value vs net after intervention cost.

## API

```http
POST /learn-policy-rules
```

```json
{
  "rows": [{"y": 2.1, "t": 1, "baseline_yield": 1.0, "slope_degrees": 5.0}],
  "outcome": "y",
  "treatment": "t",
  "covariates": ["baseline_yield", "slope_degrees"],
  "learner": "tree",
  "recommended_treatment_label": "treat_with_shade_trees",
  "budget": 50000
}
```

## CLI

```bash
PYTHONPATH=src python scripts/learn_targeting_rules.py \
  --panel data/raw/farm_panel.parquet \
  --outcome yield_delta \
  --treatment treated \
  --covariates baseline_yield,slope_degrees \
  --region ghana \
  --budget 100000
```

Report: `reports/targeting/policy_rules_{region}_{date}.md`

## Python

```python
from analysis.policy_targeting import learn_policy_tree, render_policy_rules

result = learn_policy_tree(df, treatment_col="t", outcome_col="y", covariate_cols=["x0"])
print(render_policy_rules(result, recommended_treatment_label="treat_with_shade_trees"))
```
