# Cocoa quality model

`models.cocoa_quality.CocoaQualityModel` predicts three commercial quality signals from farm and harvest-window covariates:

- `fermentation_index` (0-1): proxy for bean fermentation completeness and consistency.
- `defect_rate` (%): expected mold/slaty/flat/other defect share.
- `fine_flavor_probability` (0-1): probability that the lot clears a fine-flavor premium threshold.

Inputs include yield, harvest-window precipitation, Q3/Q4 heat-stress days, shade cover, fermentation practice, drying method, and farm age. `scripts/train_cocoa_quality.py` starts with synthetic labels and is intentionally marked for ICCO/cooperative real-data integration once quality-lab labels are available.

`POST /simulate-intervention` accepts `include_quality=true`; the response includes a `quality` block and the financial valuation uses `config/quality_premiums.yaml` to apply premium/penalty adjustments such as fine-flavor upside and high-defect discounts.

References: ICCO fine/flavour cocoa panel standards, cut-test defect conventions, and cooperative buyer premium schedules.
