"""
Policy targeting utilities for cooperative-level rollout planning.

Includes greedy CATE ranking and honest doubly-robust policy trees/forests
(Athey & Wager 2021; EconML ``DRPolicyTree`` / ``DRPolicyForest``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from analysis.heterogeneity import CATEResult, default_nuisance_models, estimate_cate

try:
    from econml.policy import DRPolicyForest, DRPolicyTree, PolicyTree
except ImportError as exc:  # pragma: no cover
    DRPolicyTree = None  # type: ignore[misc, assignment]
    DRPolicyForest = None  # type: ignore[misc, assignment]
    PolicyTree = None  # type: ignore[misc, assignment]
    _ECONML_IMPORT_ERROR = exc
else:
    _ECONML_IMPORT_ERROR = None

PS_CLIP = (0.01, 0.99)
Z_975 = 1.96
DEFAULT_BOOTSTRAP_REPS = 100


def _require_econml() -> None:
    if _ECONML_IMPORT_ERROR is not None:
        raise ImportError(
            "Policy tree learning requires econml>=0.15. Install with: pip install 'econml>=0.15'"
        ) from _ECONML_IMPORT_ERROR


def _check_binary_treatment(series: pd.Series) -> None:
    vals = set(pd.unique(series.dropna()))
    if not vals.issubset({0, 1, False, True}):
        raise ValueError("treatment_col must be binary 0/1 (control/treated)")


def _panel_arrays(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [c for c in covariate_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariate columns: {missing}")
    if treatment_col not in df.columns or outcome_col not in df.columns:
        raise ValueError("Missing treatment or outcome column")
    x = df[list(covariate_cols)].astype(float).to_numpy()
    if np.isnan(x).any():
        raise ValueError("Covariates contain missing values; impute first")
    y = df[outcome_col].astype(float).to_numpy()
    t = df[treatment_col].astype(int).to_numpy()
    _check_binary_treatment(df[treatment_col])
    return x, y, t


@dataclass(frozen=True)
class PolicyTreeResult:
    """Fitted honest DR policy tree with evaluation metadata."""

    tree: Any
    feature_names: list[str]
    treatment_names: list[str]
    policy_value_estimate: float
    policy_value_ci: tuple[float, float]
    leaf_summary: pd.DataFrame
    greedy_policy_value: float | None
    cost_aware: bool


@dataclass(frozen=True)
class PolicyForestResult:
    """Fitted honest DR policy forest (rules exported from tree_id=0)."""

    forest: Any
    feature_names: list[str]
    treatment_names: list[str]
    policy_value_estimate: float
    policy_value_ci: tuple[float, float]
    leaf_summary: pd.DataFrame
    greedy_policy_value: float | None
    cost_aware: bool
    n_estimators: int
    tree_id_for_rules: int = 0


def rank_farms_by_uplift(
    cate_result: CATEResult,
    *,
    intervention_cost_usd_per_farm: float,
    cocoa_price_usd: float,
    farm_areas_ha: pd.Series,
) -> pd.DataFrame:
    """
    Rank farms by expected net uplift USD.

    Net uplift (USD) = max(tau_hat, 0) * area_ha * cocoa_price_usd - intervention_cost
    Ties are broken by lower standard error (more certain uplift first).
    """
    tau = cate_result.tau_hat
    se = cate_result.se.reindex(tau.index)
    area = farm_areas_ha.reindex(tau.index).astype(float)

    avoided_tonnes = np.maximum(tau.to_numpy(dtype=float), 0.0) * area.to_numpy(dtype=float)
    gross_usd = avoided_tonnes * float(cocoa_price_usd)
    net_usd = gross_usd - float(intervention_cost_usd_per_farm)

    out = pd.DataFrame(
        {
            "tau_hat_tonnes_per_ha": tau.astype(float),
            "se": se.astype(float),
            "area_ha": area,
            "avoided_loss_tonnes": avoided_tonnes,
            "gross_uplift_usd": gross_usd,
            "net_uplift_usd": net_usd,
        },
        index=tau.index,
    )
    return out.sort_values(by=["net_uplift_usd", "se"], ascending=[False, True])


def policy_value_curve(
    ranked: pd.DataFrame,
    *,
    uplift_col: str = "avoided_loss_tonnes",
) -> pd.DataFrame:
    """
    Cumulative value curve for targeting the top-K farms.

    Parameters
    ----------
    ranked:
        Output from :func:`rank_farms_by_uplift` (already sorted).
    uplift_col:
        Column to cumulate, e.g. ``avoided_loss_tonnes`` or ``net_uplift_usd``.
    """
    if uplift_col not in ranked.columns:
        raise ValueError(f"Missing uplift column '{uplift_col}'")
    vals = ranked[uplift_col].to_numpy(dtype=float)
    cum = np.cumsum(np.maximum(vals, 0.0))
    return pd.DataFrame(
        {
            "k": np.arange(1, len(ranked) + 1),
            "cumulative_value": cum,
        }
    )


def optimal_targeting_policy(
    cate_estimates: pd.Series | np.ndarray,
    costs_per_farm: pd.Series | np.ndarray,
    budget: float,
) -> np.ndarray:
    """
    Greedy budgeted targeting by CATE / cost ratio.

    Returns a boolean mask (same length as inputs) indicating selected units.
    """
    tau = np.asarray(cate_estimates, dtype=float).ravel()
    costs = np.asarray(costs_per_farm, dtype=float).ravel()
    if len(tau) != len(costs):
        raise ValueError("cate_estimates and costs_per_farm must have the same length")
    if budget <= 0:
        return np.zeros(len(tau), dtype=bool)

    safe_costs = np.where(costs > 0, costs, np.inf)
    score = tau / safe_costs
    order = np.argsort(-score)
    selected = np.zeros(len(tau), dtype=bool)
    spent = 0.0
    for i in order:
        c = costs[i]
        if not np.isfinite(c) or c <= 0:
            continue
        if spent + c <= budget:
            selected[i] = True
            spent += c
    return selected


def _crossfit_nuisances_policy(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted outcome and propensity nuisances for DR policy evaluation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    m_hat = np.zeros_like(y, dtype=float)
    e_hat = np.zeros_like(t, dtype=float)
    reg, clf = default_nuisance_models(random_state)
    for tr, te in kf.split(x):
        m = clone(reg)
        g = clone(clf)
        m.fit(x[tr], y[tr])
        m_hat[te] = m.predict(x[te])
        g.fit(x[tr], t[tr])
        e_hat[te] = g.predict_proba(x[te])[:, 1]
    e_hat = np.clip(e_hat, PS_CLIP[0], PS_CLIP[1])
    return m_hat, e_hat


def doubly_robust_policy_value(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    policy_mask: np.ndarray | pd.Series,
    tau_hat: np.ndarray | pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> float:
    """
    Doubly-robust estimate of the policy value for targeting rule ``policy_mask``.

    V(pi) = E[ pi(X) * tau(X) ] + E[ (pi*T - pi*e) / (e(1-e)) * (Y - m - tau*T) ]
    """
    pi = np.asarray(policy_mask, dtype=float).ravel()
    tau = np.asarray(tau_hat, dtype=float).ravel()
    y = df[outcome_col].astype(float).to_numpy()
    t = df[treatment_col].astype(int).to_numpy()
    x = df[list(covariate_cols)].astype(float).to_numpy()
    if len(pi) != len(y) or len(tau) != len(y):
        raise ValueError("policy_mask and tau_hat must align with df rows")

    m_hat, e_hat = _crossfit_nuisances_policy(
        x, y, t, n_folds=n_folds, random_state=random_state
    )
    direct = pi * tau
    resid = y - m_hat - tau * t
    ipw_correction = (pi * t - pi * e_hat) / (e_hat * (1.0 - e_hat)) * resid
    return float(np.mean(direct + ipw_correction))


def _bootstrap_policy_value_ci(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    tree: Any,
    n_bootstrap: int,
    n_folds: int,
    random_state: int,
) -> tuple[float, float]:
    """Fixed-policy bootstrap CI for DR policy value (tree policy frozen)."""
    x = df[list(covariate_cols)].astype(float).to_numpy()
    n = len(df)
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        x_b = x[idx]
        pi_b = (tree.predict(x_b) == 1).astype(float)
        tau_b = tree.predict_value(x_b).ravel()
        estimates.append(
            doubly_robust_policy_value(
                df_b,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                covariate_cols=covariate_cols,
                policy_mask=pi_b,
                tau_hat=tau_b,
                n_folds=n_folds,
                random_state=random_state + b + 1,
            )
        )
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def _greedy_policy_value(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    cost_col: str | None,
    intervention_cost_usd_per_farm: float,
    budget: float | None,
    n_folds: int,
    random_state: int,
    cate_method: str = "r_learner",
) -> float | None:
    if budget is None or budget <= 0:
        return None
    try:
        cate = estimate_cate(
            df,
            outcome=outcome_col,
            treatment=treatment_col,
            covariates=list(covariate_cols),
            method=cate_method,  # type: ignore[arg-type]
            n_folds=n_folds,
            random_state=random_state,
        )
    except Exception:
        return None
    if cost_col is not None and cost_col in df.columns:
        costs = df[cost_col].astype(float)
    else:
        costs = pd.Series(float(intervention_cost_usd_per_farm), index=cate.tau_hat.index)
    mask, _ = targeting_from_cate(cate, costs, budget)
    tau = cate.tau_hat.to_numpy(dtype=float)
    return doubly_robust_policy_value(
        df.reset_index(drop=True),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        policy_mask=mask,
        tau_hat=tau,
        n_folds=n_folds,
        random_state=random_state,
    )


def _fit_dr_policy_tree(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    max_depth: int,
    min_samples_leaf: int,
    n_folds: int,
    random_state: int,
) -> Any:
    _require_econml()
    reg, clf = default_nuisance_models(random_state)
    tree = DRPolicyTree(
        model_regression=reg,
        model_propensity=clf,
        honest=True,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        cv=n_folds,
        categories=[0, 1],
        random_state=random_state,
    )
    tree.fit(y, t, X=x)
    return tree


@dataclass
class _CostAdjustedPolicyView:
    """Wraps a cost-adjusted honest PolicyTree plus DR nuisance estimates."""

    policy_model_: Any
    _dr_tree: Any

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.policy_model_.predict(x)

    def predict_value(self, x: np.ndarray) -> np.ndarray:
        return self._dr_tree.predict_value(x)

    def policy_feature_names(self, *, feature_names: list[str] | None = None) -> list[str]:
        return list(self._dr_tree.policy_feature_names(feature_names=feature_names))

    def policy_treatment_names(self, *, treatment_names: list[str] | None = None) -> list[str]:
        return list(self._dr_tree.policy_treatment_names(treatment_names=treatment_names))


def _fit_cost_aware_policy_view(
    tree: Any,
    x: np.ndarray,
    cost: np.ndarray,
    *,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> _CostAdjustedPolicyView:
    """
    Second-stage honest policy tree on DR rewards minus cost.

    Trade-off: nuisances come from the initial DRPolicyTree fit; the relabeled
    welfare matrix optimizes net benefit (uplift − cost) rather than raw CATE.
    """
    _require_econml()
    reward_treat = np.asarray(tree.predict_value(x), dtype=float).reshape(-1) - cost
    y_policy = np.column_stack([np.zeros(len(x)), reward_treat])
    new_pm = PolicyTree(
        honest=True,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    new_pm.fit(x, y_policy)
    return _CostAdjustedPolicyView(policy_model_=new_pm, _dr_tree=tree)


def _leaf_node_for_row(x_row: np.ndarray, tree_struct: Any) -> int:
    node = 0
    while tree_struct.feature[node] >= 0:
        feat = int(tree_struct.feature[node])
        if x_row[feat] <= tree_struct.threshold[node]:
            node = int(tree_struct.children_left[node])
        else:
            node = int(tree_struct.children_right[node])
    return int(node)


def _leaf_ids_from_tree(policy_model: Any, x: np.ndarray) -> np.ndarray:
    tree_struct = policy_model.tree_
    return np.array([_leaf_node_for_row(x[i], tree_struct) for i in range(x.shape[0])], dtype=int)


def _build_leaf_summary(
    df: pd.DataFrame,
    tree: Any,
    x: np.ndarray,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    feature_names: list[str],
    rule_texts: dict[int, str],
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    pm = tree.policy_model_
    leaf_ids = _leaf_ids_from_tree(pm, x)
    treat_rec = (tree.predict(x) == 1).astype(int)
    uplift = np.asarray(tree.predict_value(x), dtype=float).reshape(-1)
    rng = np.random.default_rng(random_state)

    rows: list[dict[str, Any]] = []
    for leaf_id in np.unique(leaf_ids):
        mask = np.asarray(leaf_ids == leaf_id, dtype=bool)
        n_units = int(mask.sum())
        if n_units == 0:
            continue
        u_leaf = uplift[mask]
        mean_uplift = float(np.mean(u_leaf))
        if n_bootstrap <= 0:
            ci_low = ci_high = mean_uplift
        else:
            boot_means = []
            for _ in range(n_bootstrap):
                idx = rng.choice(np.where(mask)[0], size=n_units, replace=True)
                boot_means.append(float(np.mean(uplift[idx])))
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        rows.append(
            {
                "leaf_id": int(leaf_id),
                "rule_text": rule_texts.get(int(leaf_id), ""),
                "n_units": n_units,
                "treat_fraction": float(np.mean(treat_rec[mask])),
                "expected_uplift": mean_uplift,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)


def _tree_node_treatment_label(
    node: int,
    tree_struct: Any,
    *,
    treat_label: str,
) -> str:
    values = tree_struct.value[node].ravel()
    if len(values) >= 2 and values[1] > values[0]:
        return treat_label
    return "do_not_treat"


def _collect_leaf_rules(
    node: int,
    tree_struct: Any,
    feature_names: list[str],
    conditions: list[str],
    *,
    treat_label: str,
    rules: dict[int, str],
) -> None:
    if tree_struct.feature[node] < 0:
        cond = " AND ".join(conditions) if conditions else "TRUE"
        action = _tree_node_treatment_label(node, tree_struct, treat_label=treat_label)
        rules[int(node)] = f"IF {cond} THEN {action}"
        return
    feat_idx = int(tree_struct.feature[node])
    name = feature_names[feat_idx]
    thresh = float(tree_struct.threshold[node])
    left = int(tree_struct.children_left[node])
    right = int(tree_struct.children_right[node])
    _collect_leaf_rules(
        left,
        tree_struct,
        feature_names,
        conditions + [f"{name} <= {thresh:.4g}"],
        treat_label=treat_label,
        rules=rules,
    )
    _collect_leaf_rules(
        right,
        tree_struct,
        feature_names,
        conditions + [f"{name} > {thresh:.4g}"],
        treat_label=treat_label,
        rules=rules,
    )


def render_policy_rules(
    result: PolicyTreeResult,
    *,
    recommended_treatment_label: str = "treat",
) -> list[str]:
    """
    Human-readable if-then-else rules using original covariate names only.
    """
    pm = result.tree.policy_model_
    tree_struct = pm.tree_
    rules_map: dict[int, str] = {}
    _collect_leaf_rules(
        0,
        tree_struct,
        result.feature_names,
        [],
        treat_label=recommended_treatment_label,
        rules=rules_map,
    )
    leaf_ids = sorted(rules_map.keys())
    return [rules_map[lid] for lid in leaf_ids]


def render_policy_rules_from_forest(
    result: PolicyForestResult,
    *,
    tree_id: int = 0,
    recommended_treatment_label: str = "treat",
) -> list[str]:
    """Export rules from a single forest tree (default ``tree_id=0``)."""
    sub = result.forest.policy_model_[tree_id]
    tree_struct = sub.tree_
    rules_map: dict[int, str] = {}
    _collect_leaf_rules(
        0,
        tree_struct,
        result.feature_names,
        [],
        treat_label=recommended_treatment_label,
        rules=rules_map,
    )
    return [rules_map[k] for k in sorted(rules_map.keys())]


def _evaluate_tree_policy(
    df: pd.DataFrame,
    tree: Any,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    n_folds: int,
    random_state: int,
    n_bootstrap: int,
) -> tuple[float, tuple[float, float], np.ndarray, np.ndarray]:
    x, _, _ = _panel_arrays(df, treatment_col=treatment_col, outcome_col=outcome_col, covariate_cols=covariate_cols)
    pi = (tree.predict(x) == 1).astype(float)
    tau = tree.predict_value(x).ravel()
    pv = doubly_robust_policy_value(
        df.reset_index(drop=True),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        policy_mask=pi,
        tau_hat=tau,
        n_folds=n_folds,
        random_state=random_state,
    )
    if n_bootstrap <= 0:
        half = 0.05 * max(abs(pv), 1e-6)
        ci = (pv - half, pv + half)
    else:
        ci = _bootstrap_policy_value_ci(
            df.reset_index(drop=True),
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            covariate_cols=covariate_cols,
            tree=tree,
            n_bootstrap=n_bootstrap,
            n_folds=n_folds,
            random_state=random_state,
        )
    return pv, (ci[0], ci[1]), pi, tau


def learn_policy_tree(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    max_depth: int = 4,
    min_samples_leaf: int = 50,
    cost_col: str | None = None,
    n_folds: int = 5,
    random_state: int = 42,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    intervention_cost_usd_per_farm: float = 0.0,
    budget: float | None = None,
    recommended_treatment_label: str = "treat",
    cate_method: str = "r_learner",
) -> PolicyTreeResult:
    """
    Learn an honest DR policy tree for binary treatment targeting.

    When ``cost_col`` is set, a second-stage refit maximizes net benefit
    (estimated uplift minus per-unit cost) instead of raw CATE. Greedy baselines
    with a budget use ``tau / cost`` via :func:`targeting_from_cate`.
    """
    x, y, t = _panel_arrays(
        df, treatment_col=treatment_col, outcome_col=outcome_col, covariate_cols=covariate_cols
    )
    tree = _fit_dr_policy_tree(
        x, y, t,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_folds=n_folds,
        random_state=random_state,
    )
    cost_aware = cost_col is not None
    policy_est = tree
    if cost_aware:
        if cost_col not in df.columns:
            raise ValueError(f"Missing cost column '{cost_col}'")
        cost = df[cost_col].astype(float).to_numpy()
        policy_est = _fit_cost_aware_policy_view(
            tree, x, cost,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    feature_names = list(
        policy_est.policy_feature_names(feature_names=list(covariate_cols))
    )
    treatment_names = list(policy_est.policy_treatment_names())
    df_reset = df.reset_index(drop=True)

    pv, ci, _, _ = _evaluate_tree_policy(
        df_reset,
        policy_est,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        n_folds=n_folds,
        random_state=random_state,
        n_bootstrap=n_bootstrap,
    )

    pm = policy_est.policy_model_
    rules_by_leaf: dict[int, str] = {}
    _collect_leaf_rules(
        0, pm.tree_, feature_names, [], treat_label=recommended_treatment_label, rules=rules_by_leaf
    )
    leaf_summary = _build_leaf_summary(
        df_reset,
        policy_est,
        x,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        feature_names=feature_names,
        rule_texts=rules_by_leaf,
        n_bootstrap=max(50, n_bootstrap) if n_bootstrap > 0 else 0,
        random_state=random_state,
    )

    greedy_pv = _greedy_policy_value(
        df_reset,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        cost_col=cost_col,
        intervention_cost_usd_per_farm=intervention_cost_usd_per_farm,
        budget=budget,
        n_folds=n_folds,
        random_state=random_state,
        cate_method=cate_method,
    )

    return PolicyTreeResult(
        tree=policy_est,
        feature_names=feature_names,
        treatment_names=treatment_names,
        policy_value_estimate=pv,
        policy_value_ci=ci,
        leaf_summary=leaf_summary,
        greedy_policy_value=greedy_pv,
        cost_aware=cost_aware,
    )


def learn_policy_forest(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    max_depth: int = 4,
    min_samples_leaf: int = 50,
    cost_col: str | None = None,
    n_estimators: int = 500,
    n_folds: int = 5,
    random_state: int = 42,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    intervention_cost_usd_per_farm: float = 0.0,
    budget: float | None = None,
    recommended_treatment_label: str = "treat",
    tree_id_for_rules: int = 0,
    cate_method: str = "r_learner",
) -> PolicyForestResult:
    """
    Learn an honest DR policy forest (default 500 trees).

    Interpretable rules are taken from ``tree_id_for_rules`` (default 0).
    Honest estimation is enabled (Athey & Wager 2021).
    """
    _require_econml()
    x, y, t = _panel_arrays(
        df, treatment_col=treatment_col, outcome_col=outcome_col, covariate_cols=covariate_cols
    )
    reg, clf = default_nuisance_models(random_state)
    forest = DRPolicyForest(
        model_regression=reg,
        model_propensity=clf,
        honest=True,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        cv=n_folds,
        categories=[0, 1],
        random_state=random_state,
        n_jobs=-1,
    )
    forest.fit(y, t, X=x)

    cost_aware = cost_col is not None
    policy_est: Any = forest
    if cost_aware:
        if cost_col not in df.columns:
            raise ValueError(f"Missing cost column '{cost_col}'")
        cost = df[cost_col].astype(float).to_numpy()
        reward_treat = np.asarray(forest.predict_value(x), dtype=float).reshape(-1) - cost
        y_policy = np.column_stack([np.zeros(len(x)), reward_treat])
        new_sub = PolicyTree(
            honest=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        new_sub.fit(x, y_policy)
        policy_est = _CostAdjustedPolicyView(policy_model_=new_sub, _dr_tree=forest)

    feature_names = list(forest.policy_feature_names(feature_names=list(covariate_cols)))
    treatment_names = list(forest.policy_treatment_names())
    df_reset = df.reset_index(drop=True)

    pv, ci, _, _ = _evaluate_tree_policy(
        df_reset,
        policy_est,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        n_folds=n_folds,
        random_state=random_state,
        n_bootstrap=n_bootstrap,
    )

    sub_tree = policy_est.policy_model_
    rules_by_leaf: dict[int, str] = {}
    _collect_leaf_rules(
        0,
        sub_tree.tree_,
        feature_names,
        [],
        treat_label=recommended_treatment_label,
        rules=rules_by_leaf,
    )
    leaf_summary = _build_leaf_summary(
        df_reset,
        policy_est,
        x,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        feature_names=feature_names,
        rule_texts=rules_by_leaf,
        n_bootstrap=max(50, n_bootstrap) if n_bootstrap > 0 else 0,
        random_state=random_state,
    )

    greedy_pv = _greedy_policy_value(
        df_reset,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        cost_col=cost_col,
        intervention_cost_usd_per_farm=intervention_cost_usd_per_farm,
        budget=budget,
        n_folds=n_folds,
        random_state=random_state,
        cate_method=cate_method,
    )

    return PolicyForestResult(
        forest=forest,
        feature_names=feature_names,
        treatment_names=treatment_names,
        policy_value_estimate=pv,
        policy_value_ci=ci,
        leaf_summary=leaf_summary,
        greedy_policy_value=greedy_pv,
        cost_aware=cost_aware,
        n_estimators=n_estimators,
        tree_id_for_rules=tree_id_for_rules,
    )


def targeting_from_cate(
    cate_result: CATEResult,
    costs_per_farm: pd.Series,
    budget: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build greedy targeting mask and ranked table from :class:`CATEResult`.

    Returns ``(policy_mask, ranked_df)`` where ``ranked_df`` includes
    ``selected`` and ``cate_per_cost`` columns.
    """
    tau = cate_result.tau_hat
    costs = costs_per_farm.reindex(tau.index).astype(float)
    mask = optimal_targeting_policy(tau, costs, budget)
    safe_costs = costs.replace(0, np.nan)
    ranked = pd.DataFrame(
        {
            "tau_hat": tau,
            "cost": costs,
            "cate_per_cost": tau / safe_costs,
            "selected": mask,
        },
        index=tau.index,
    )
    ranked = ranked.sort_values(
        ["selected", "cate_per_cost"],
        ascending=[False, False],
        na_position="last",
    )
    return mask, ranked


def first_split_threshold(
    result: PolicyTreeResult,
    feature_name: str,
) -> float | None:
    """Return the root split threshold on ``feature_name`` (for validation tests)."""
    pm = getattr(result.tree, "policy_model_", result.tree)
    if not hasattr(pm, "tree_"):
        pm = result.tree.policy_model_
    tree_struct = pm.tree_
    if tree_struct.feature[0] < 0:
        return None
    idx = result.feature_names.index(feature_name)
    if int(tree_struct.feature[0]) != idx:
        return None
    return float(tree_struct.threshold[0])


def root_split_feature(result: PolicyTreeResult) -> str | None:
    """Feature name used at the tree root, if any."""
    pm = getattr(result.tree, "policy_model_", None) or result.tree
    if not hasattr(pm, "tree_"):
        pm = result.tree.policy_model_
    tree_struct = pm.tree_
    if tree_struct.feature[0] < 0:
        return None
    feat_idx = int(tree_struct.feature[0])
    if feat_idx < 0 or feat_idx >= len(result.feature_names):
        return None
    return result.feature_names[feat_idx]


__all__ = [
    "PolicyTreeResult",
    "PolicyForestResult",
    "rank_farms_by_uplift",
    "policy_value_curve",
    "optimal_targeting_policy",
    "doubly_robust_policy_value",
    "targeting_from_cate",
    "learn_policy_tree",
    "learn_policy_forest",
    "render_policy_rules",
    "render_policy_rules_from_forest",
    "first_split_threshold",
    "root_split_feature",
]
