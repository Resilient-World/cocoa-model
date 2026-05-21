"""
Borusyak, Jaravel & Spiess (2024) imputation estimator for staggered DiD.

Two-way fixed effects imputation of untreated outcomes, then ATT aggregation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis._staggered_did_common import (
    Z_975,
    normal_ci,
    prepare_staggered_panel,
)


@dataclass
class BJSResult:
    """BJS imputation ATT."""

    att: float
    se: float
    ci_low: float
    ci_high: float
    pretrend_stat: float | None = None
    pretrend_pvalue: float | None = None
    n_treated: int = 0
    method: str = "bjs_imputation"


class BorusyakJaravelSpiess:
    """
    BJS (2024) imputation estimator: fit Y on unit + time FE (and covariates) on
    not-yet-treated / never-treated rows, impute Y(0), aggregate tau_it.
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,
        *,
        unit_col: str = "farm_id",
        time_col: str = "period",
        treat_time_col: str = "treatment_period",
        outcome_col: str = "yield",
        covariate_cols: Sequence[str] | None = None,
        weights: str = "ATT",
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.panel = prepare_staggered_panel(
            panel_df,
            unit_col=unit_col,
            time_col=time_col,
            treat_time_col=treat_time_col,
            outcome_col=outcome_col,
            covariate_cols=covariate_cols,
        )
        self.weights = weights
        self.alpha = alpha
        self.random_state = random_state

    def _estimation_sample_mask(self) -> np.ndarray:
        """Never-treated or not-yet-treated (D_it = 0 for BJS Step 1)."""
        g = self.panel.df["_G"].to_numpy(dtype=float)
        t = self.panel.df[self.panel.time_col].to_numpy(dtype=float)
        never = np.isnan(g)
        not_yet = (~never) & (t < g)
        return never | not_yet

    def _fit_y0(self) -> pd.Series:
        """Impute untreated outcome Y_it(0) for all rows."""
        from linearmodels.panel import PanelOLS

        work = self.panel.df.copy()
        est_mask = self._estimation_sample_mask()
        est = work.loc[est_mask].set_index([self.panel.unit_col, self.panel.time_col])
        y = est[self.panel.outcome_col]
        full = work.set_index([self.panel.unit_col, self.panel.time_col])
        if self.panel.covariate_cols:
            exog = est[self.panel.covariate_cols]
            mod = PanelOLS(y, exog, entity_effects=True, time_effects=True)
            res = mod.fit()
            y0 = res.predict(exog=full[self.panel.covariate_cols])
        else:
            const = pd.DataFrame(
                {"const": 1.0},
                index=est.index,
            )
            mod = PanelOLS(y, const, entity_effects=True, time_effects=True)
            res = mod.fit()
            const_all = pd.DataFrame({"const": 1.0}, index=full.index)
            y0 = res.predict(exog=const_all)
        y0_arr = np.asarray(y0).reshape(-1)
        return pd.Series(y0_arr, index=full.index)

    def _row_weights(self) -> np.ndarray:
        """ATT weights: equal weight on treated observations (D_it=1)."""
        g = self.panel.df["_G"].to_numpy(dtype=float)
        t = self.panel.df[self.panel.time_col].to_numpy(dtype=float)
        treated_now = (~np.isnan(g)) & (t >= g)
        if self.weights.upper() == "ATT":
            w = treated_now.astype(float)
            s = w.sum()
            return w / s if s > 0 else w
        raise ValueError(f"Unknown weights scheme: {self.weights!r}")

    def estimate(self) -> BJSResult:
        """Run full BJS pipeline and return ATT with analytic cluster SE."""
        y0_hat = self._fit_y0()
        work = self.panel.df.set_index([self.panel.unit_col, self.panel.time_col])
        y = work[self.panel.outcome_col]
        tau = y - y0_hat
        tau = tau.reindex(work.index)

        w = self._row_weights()
        tau_vals = tau.to_numpy(dtype=float)
        att = float(np.nansum(w * tau_vals))

        g = work["_G"].to_numpy(dtype=float)
        t = work.index.get_level_values(1).to_numpy(dtype=float)
        treated_now = (~np.isnan(g)) & (t >= g)
        score = w * (tau_vals - att)
        units = work.index.get_level_values(0).to_numpy()
        cluster_sums: dict[object, float] = {}
        for sc, u, tr in zip(score, units, treated_now):
            if not tr or np.isnan(sc):
                continue
            cluster_sums[u] = cluster_sums.get(u, 0.0) + float(sc)
        vals = np.array(list(cluster_sums.values()))
        se = float(np.sqrt(np.sum(vals**2))) if len(vals) >= 2 else float("nan")
        lo, hi = normal_ci(att, se, self.alpha)

        pretrend_stat, pretrend_p = self._pretrend_test(tau, work)

        return BJSResult(
            att=att,
            se=se,
            ci_low=lo,
            ci_high=hi,
            pretrend_stat=pretrend_stat,
            pretrend_pvalue=pretrend_p,
            n_treated=int(treated_now.sum()),
        )

    def _pretrend_test(
        self,
        tau: pd.Series,
        work: pd.DataFrame,
    ) -> tuple[float | None, float | None]:
        """Roth (2022)-style pre-trend test on imputation residuals (pre-treatment periods)."""
        g = work["_G"].to_numpy(dtype=float)
        t = work.index.get_level_values(1).to_numpy(dtype=float)
        pre = (~np.isnan(g)) & (t < g)
        if pre.sum() < 3:
            return None, None
        resid = tau.to_numpy()[pre]
        resid = resid[~np.isnan(resid)]
        if len(resid) < 3:
            return None, None
        mean_r = float(resid.mean())
        se_r = float(resid.std(ddof=1) / np.sqrt(len(resid)))
        if se_r <= 0:
            return mean_r, None
        z = mean_r / se_r
        from math import erf, sqrt

        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
        return mean_r, float(p)


__all__ = ["BJSResult", "BorusyakJaravelSpiess"]
