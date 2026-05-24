from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.causal_discovery import (
    discover_dag_notears,
    discover_dag_pc,
    ensemble_discovered_dag,
)


def _edge_f1(found: set[frozenset[str]], truth: set[frozenset[str]]) -> float:
    tp = len(found & truth)
    precision = tp / max(1, len(found))
    recall = tp / max(1, len(truth))
    return 2.0 * precision * recall / max(1e-9, precision + recall)


def _assert_skeleton(method, df: pd.DataFrame, truth: set[frozenset[str]]) -> None:
    dag = method(df)
    found = {frozenset(edge) for edge in dag}
    assert _edge_f1(found, truth) >= 0.9


def test_pc_and_notears_recover_chain_skeleton() -> None:
    rng = np.random.default_rng(1)
    n = 800
    a = rng.normal(size=n)
    b = 1.2 * a + rng.normal(scale=0.2, size=n)
    c = -0.8 * b + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"A": a, "B": b, "C": c})
    truth = {frozenset(("A", "B")), frozenset(("B", "C"))}
    _assert_skeleton(discover_dag_pc, df, truth)
    _assert_skeleton(discover_dag_notears, df, truth)


def test_pc_and_notears_recover_fork_skeleton() -> None:
    rng = np.random.default_rng(2)
    n = 800
    a = rng.normal(size=n)
    b = 1.0 * a + rng.normal(scale=0.2, size=n)
    c = -1.1 * a + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"A": a, "B": b, "C": c})
    truth = {frozenset(("A", "B")), frozenset(("A", "C"))}
    _assert_skeleton(discover_dag_pc, df, truth)
    _assert_skeleton(discover_dag_notears, df, truth)


def test_pc_and_notears_recover_collider_skeleton() -> None:
    rng = np.random.default_rng(3)
    n = 800
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = 0.9 * a - 1.0 * b + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"A": a, "B": b, "C": c})
    truth = {frozenset(("A", "C")), frozenset(("B", "C"))}
    _assert_skeleton(discover_dag_pc, df, truth)
    _assert_skeleton(discover_dag_notears, df, truth)


def test_ensemble_confidence_fraction() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.2, size=200)
    df = pd.DataFrame({"A": a, "B": b})
    dag = ensemble_discovered_dag(df, methods=("pc", "notears"))
    assert dag[("A", "B")] == 1.0
