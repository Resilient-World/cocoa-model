# Causal DAG validation

## Assumed DAG
- shade_trees → microclimate_index
- microclimate_index → cssvd_prevalence_delta
- cssvd_prevalence_delta → yield
- microclimate_index → yield

## PC discovered DAG
- shade_trees → microclimate_index (0.62)
- microclimate_index → cssvd_prevalence_delta (0.76)
- microclimate_index → yield (0.28)
- cssvd_prevalence_delta → yield (0.51)

## NOTEARS-MLP discovered DAG
- shade_trees → microclimate_index (0.62)
- microclimate_index → cssvd_prevalence_delta (0.76)
- cssvd_prevalence_delta → yield (0.51)

## GES discovered DAG
- shade_trees → microclimate_index (0.62)
- microclimate_index → cssvd_prevalence_delta (0.76)
- microclimate_index → yield (0.28)
- cssvd_prevalence_delta → yield (0.51)

## Ensemble confidence DAG
- cssvd_prevalence_delta → yield (1.00)
- microclimate_index → cssvd_prevalence_delta (1.00)
- microclimate_index → yield (0.67)
- shade_trees → microclimate_index (1.00)

## DAGComparisonReport
```json
{
  "edges_in_both": [
    [
      "cssvd_prevalence_delta",
      "yield"
    ],
    [
      "microclimate_index",
      "cssvd_prevalence_delta"
    ],
    [
      "microclimate_index",
      "yield"
    ],
    [
      "shade_trees",
      "microclimate_index"
    ]
  ],
  "edges_only_discovered": [],
  "edges_only_assumed": [],
  "hamming_distance": 0,
  "structural_hamming_distance": 0,
  "discovered_edge_count": 4,
  "assumed_edge_count": 4,
  "metadata": {}
}
```

## Recommendation
ASSUMED DAG VALIDATED
