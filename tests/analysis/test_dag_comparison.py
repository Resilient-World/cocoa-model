from __future__ import annotations

import json

from analysis.causal_discovery import compare_with_assumed_dag


def test_compare_with_assumed_dag_fields(tmp_path) -> None:
    assumed = tmp_path / "assumed.json"
    assumed.write_text(
        json.dumps({"edges": [["A", "B"], ["B", "C"], ["C", "D"]]}),
        encoding="utf-8",
    )
    discovered = {("A", "B"): 1.0, ("B", "D"): 0.7, ("D", "C"): 0.3}
    report = compare_with_assumed_dag(discovered, assumed)
    assert report.edges_in_both == [("A", "B")]
    assert report.edges_only_discovered == [("B", "D"), ("D", "C")]
    assert report.edges_only_assumed == [("B", "C"), ("C", "D")]
    assert report.hamming_distance == 4
    assert report.structural_hamming_distance == 3
    assert report.to_dict()["discovered_edge_count"] == 3
