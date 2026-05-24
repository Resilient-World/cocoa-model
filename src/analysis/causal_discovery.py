"""Causal discovery utilities for mediation and teleconnection DAG validation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

Edge = tuple[str, str]
WeightedDag = dict[Edge, float]


@dataclass(frozen=True)
class DAGComparisonReport:
    """Structural comparison between a discovered DAG and an assumed DAG."""

    edges_in_both: list[Edge]
    edges_only_discovered: list[Edge]
    edges_only_assumed: list[Edge]
    hamming_distance: int
    structural_hamming_distance: int
    discovered_edge_count: int = 0
    assumed_edge_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges_in_both": [list(edge) for edge in self.edges_in_both],
            "edges_only_discovered": [list(edge) for edge in self.edges_only_discovered],
            "edges_only_assumed": [list(edge) for edge in self.edges_only_assumed],
            "hamming_distance": self.hamming_distance,
            "structural_hamming_distance": self.structural_hamming_distance,
            "discovered_edge_count": self.discovered_edge_count,
            "assumed_edge_count": self.assumed_edge_count,
            "metadata": self.metadata,
        }


ASSUMED_MEDIATION_EDGES: tuple[Edge, ...] = (
    ("shade_trees", "microclimate_index"),
    ("microclimate_index", "cssvd_prevalence_delta"),
    ("cssvd_prevalence_delta", "yield"),
    ("microclimate_index", "yield"),
)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    data = cast(
        pd.DataFrame,
        df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(),
    )
    if data.shape[1] < 2:
        raise ValueError("causal discovery requires at least two numeric columns")
    if len(data) < 5:
        raise ValueError("causal discovery requires at least five complete rows")
    return data


def _matrix_to_edges(
    matrix: np.ndarray, columns: Sequence[str], threshold: float = 0.0
) -> WeightedDag:
    edges: WeightedDag = {}
    arr = np.asarray(matrix, dtype=float)
    for i, src in enumerate(columns):
        for j, dst in enumerate(columns):
            if i == j:
                continue
            weight = float(arr[i, j])
            if abs(weight) > threshold:
                edges[(src, dst)] = abs(weight)
    return edges


def _partial_corr_edges(df: pd.DataFrame, *, threshold: float) -> WeightedDag:
    data = _clean_df(df)
    cols = list(data.columns)
    values = data.to_numpy(dtype=float)
    edges: WeightedDag = {}
    for i, src in enumerate(cols):
        for j in range(i + 1, len(cols)):
            dst = cols[j]
            others = [k for k in range(len(cols)) if k not in (i, j)]
            xi_raw = values[:, i]
            xj_raw = values[:, j]
            marginal = float(np.corrcoef(xi_raw, xj_raw)[0, 1])
            xi = xi_raw
            xj = xj_raw
            if others:
                z = values[:, others]
                xi = xi - LinearRegression().fit(z, xi).predict(z)
                xj = xj - LinearRegression().fit(z, xj).predict(z)
            corr = float(np.corrcoef(xi, xj)[0, 1])
            if (
                np.isfinite(corr)
                and np.isfinite(marginal)
                and abs(corr) >= threshold
                and abs(marginal) >= threshold
            ):
                edges[(src, dst)] = abs(corr)
    return edges


def _lagged_orientation(df: pd.DataFrame, edges: Mapping[Edge, float]) -> WeightedDag:
    cols = list(df.columns)
    ordered: WeightedDag = {}
    for (a, b), weight in edges.items():
        if cols.index(a) < cols.index(b):
            ordered[(a, b)] = float(weight)
        else:
            ordered[(b, a)] = float(weight)
    return ordered


def discover_dag_pc(
    df: pd.DataFrame,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
) -> WeightedDag:
    """Discover a DAG with causal-learn PC, falling back to deterministic partial correlation."""
    data = _clean_df(df)
    try:
        from causallearn.search.ConstraintBased.PC import pc  # type: ignore[import-not-found]

        cg = pc(data.to_numpy(dtype=float), alpha=alpha, indep_test=indep_test, verbose=False)
        graph = cg.G.graph
        edges: WeightedDag = {}
        cols = list(data.columns)
        for i, src in enumerate(cols):
            for j, dst in enumerate(cols):
                if i == j:
                    continue
                if graph[i, j] != 0 and graph[j, i] == 0:
                    edges[(src, dst)] = 1.0
                elif i < j and graph[i, j] != 0 and graph[j, i] != 0:
                    edges[(src, dst)] = 0.5
        return edges
    except Exception:
        threshold = max(0.2, min(0.35, alpha * 4.0))
        return _lagged_orientation(data, _partial_corr_edges(data, threshold=threshold))


def discover_dag_notears(
    df: pd.DataFrame,
    lambda1: float = 0.1,
    w_threshold: float = 0.3,
) -> WeightedDag:
    """Discover a DAG with gCastle NOTEARS-MLP, with a correlation fallback for py3.12 CI."""
    data = _clean_df(df)
    try:
        from castle.algorithms import NotearsNonlinear  # type: ignore[import-not-found]

        learner = NotearsNonlinear(lambda1=lambda1, w_threshold=w_threshold)
        learner.learn(data.to_numpy(dtype=float))
        matrix = cast(np.ndarray, learner.causal_matrix)
        return _matrix_to_edges(matrix, list(data.columns), threshold=0.0)
    except Exception:
        return _lagged_orientation(data, _partial_corr_edges(data, threshold=w_threshold))


def discover_dag_ges(df: pd.DataFrame, score: str = "bic") -> WeightedDag:
    """Discover a DAG with causal-learn GES, falling back to ordered correlations."""
    data = _clean_df(df)
    try:
        from causallearn.search.ScoreBased.GES import ges  # type: ignore[import-not-found]

        result = cast(dict[str, Any], ges(data.to_numpy(dtype=float), score_func=score))
        graph = cast(Any, result["G"]).graph
        return _matrix_to_edges(graph, list(data.columns), threshold=0.0)
    except Exception:
        return _lagged_orientation(data, _partial_corr_edges(data, threshold=0.2))


def ensemble_discovered_dag(
    df: pd.DataFrame,
    methods: Sequence[str] = ("pc", "notears", "ges"),
) -> WeightedDag:
    """Return edge confidence as the fraction of discovery methods that include the edge."""
    method_map: dict[str, Callable[[pd.DataFrame], WeightedDag]] = {
        "pc": discover_dag_pc,
        "notears": discover_dag_notears,
        "ges": discover_dag_ges,
    }
    discovered: list[WeightedDag] = []
    for method in methods:
        if method not in method_map:
            raise ValueError(f"unsupported discovery method: {method}")
        discovered.append(method_map[method](df))
    counts: dict[Edge, int] = {}
    for dag in discovered:
        for edge in dag:
            counts[edge] = counts.get(edge, 0) + 1
    denom = max(1, len(discovered))
    return {edge: count / denom for edge, count in sorted(counts.items())}


def _read_edges(path: Path) -> set[Edge]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_edges = payload.get("edges", payload)
    edges: set[Edge] = set()
    for edge in raw_edges:
        if isinstance(edge, Mapping):
            src = str(edge.get("source") or edge.get("src") or edge.get("from"))
            dst = str(edge.get("target") or edge.get("dst") or edge.get("to"))
        else:
            src, dst = str(edge[0]), str(edge[1])
        edges.add((src, dst))
    return edges


def compare_with_assumed_dag(
    discovered: Mapping[Edge, float] | Iterable[Edge],
    assumed_path: str | Path | None = None,
) -> DAGComparisonReport:
    """Compare discovered edges against a JSON DAG file or the default mediation DAG."""
    discovered_edges = (
        set(discovered.keys()) if isinstance(discovered, Mapping) else set(discovered)
    )
    assumed_edges = (
        _read_edges(Path(assumed_path)) if assumed_path else set(ASSUMED_MEDIATION_EDGES)
    )
    in_both = sorted(discovered_edges & assumed_edges)
    only_discovered = sorted(discovered_edges - assumed_edges)
    only_assumed = sorted(assumed_edges - discovered_edges)
    undirected_discovered = {frozenset(edge) for edge in discovered_edges}
    undirected_assumed = {frozenset(edge) for edge in assumed_edges}
    reversed_edges = sum(
        1
        for src, dst in discovered_edges
        if (dst, src) in assumed_edges and (src, dst) not in assumed_edges
    )
    shd = len(undirected_discovered ^ undirected_assumed) + reversed_edges
    return DAGComparisonReport(
        edges_in_both=in_both,
        edges_only_discovered=only_discovered,
        edges_only_assumed=only_assumed,
        hamming_distance=len(only_discovered) + len(only_assumed),
        structural_hamming_distance=shd,
        discovered_edge_count=len(discovered_edges),
        assumed_edge_count=len(assumed_edges),
    )


def edges_to_jsonable(edges: Mapping[Edge, float]) -> list[dict[str, Any]]:
    return [
        {"source": src, "target": dst, "confidence": float(conf)}
        for (src, dst), conf in sorted(edges.items())
    ]
