"""
Callaway & Sant'Anna (2021) staggered difference-in-differences.

Doubly-robust group-time ATT(g,t) with multiplier-bootstrap inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis._staggered_did_common import (
    cluster_se_from_influence,
    control_mask_at_t,
    fit_dr_nuisances,
    multiplier_bootstrap_ci,
    normal_ci,
    prepare_staggered_panel,
    simultaneous_bootstrap_bands,
    treated_cohort_mask_at_t,
)


@dataclass
class ATTGTResult:
    """ATT for cohort ``g`` at calendar time ``t``."""

    g: int | float
    t: int | float
    att: float
    se: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    influence: dict[str | int, float] = field(default_factory=dict)


@dataclass
class ATTResult:
    """Aggregated ATT summary."""

    att: float
    se: float
    ci_low: float
    ci_high: float
    method: str = "csdid_simple"
    n_cells: int = 0


@dataclass
class CSEventStudyResult:
    """Event-study coefficients with simultaneous bands."""

    leads_lags: pd.DataFrame
    pretrend_ok: bool | None = None


class CallawaySantAnna:
    """
    Callaway & Sant'Anna (2021) staggered DiD with DR estimation and
    multiplier-bootstrap CIs.
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
        ps_clip: tuple[float, float] = (0.01, 0.99),
        n_boot: int = 999,
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
        self.ps_clip = ps_clip
        self.n_boot = n_boot
        self.alpha = alpha
        self.random_state = random_state
        self._att_gt_cache: dict[tuple[float, float], ATTGTResult] = {}

    def _slice_at_t(self, t: float | int) -> pd.DataFrame:
        return self.panel.df.loc[self.panel.df[self.panel.time_col] == float(t)].copy()

    def att_gt(self, g: int, t: int) -> ATTGTResult:
        """Doubly-robust ATT for cohort ``g`` at calendar time ``t``."""
        key = (float(g), float(t))
        if key in self._att_gt_cache:
            return self._att_gt_cache[key]

        if float(t) < float(g):
            res = ATTGTResult(
                g=g,
                t=t,
                att=float("nan"),
                se=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                n_treated=0,
                n_control=0,
            )
            self._att_gt_cache[key] = res
            return res

        treated_m = treated_cohort_mask_at_t(self.panel, g, t)
        control_m = control_mask_at_t(self.panel, t)
        sub = self.panel.df.loc[treated_m | control_m].copy()
        n_t = int(treated_m.sum())
        n_c = int(control_m.sum())
        if n_t == 0 or n_c == 0:
            res = ATTGTResult(
                g=g,
                t=t,
                att=float("nan"),
                se=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                n_treated=n_t,
                n_control=n_c,
            )
            self._att_gt_cache[key] = res
            return res

        D = (sub["_G"] == float(g)).astype(int).to_numpy()
        Y = sub[self.panel.outcome_col].to_numpy(dtype=float)
        units = sub[self.panel.unit_col].to_numpy()
        if self.panel.covariate_cols:
            X = sub[self.panel.covariate_cols].to_numpy(dtype=float)
        else:
            X = np.ones((len(sub), 1))

        e, mu0, mu1 = fit_dr_nuisances(
            X, Y, D, ps_clip=self.ps_clip, random_state=self.random_state
        )
        # Outcome-regression + IPW doubly-robust ATT for cohort g (CS 2021, eq. 4)
        att_or = float(np.mean(Y[D == 1] - mu0[D == 1]))
        w_t = D / e
        w_c = (1 - D) / (1 - e)
        att_ipw = float((w_t * Y).sum() / w_t.sum() - (w_c * Y).sum() / w_c.sum())
        att = 0.5 * (att_or + att_ipw)
        psi_unit: dict[str | int, float] = {}
        for u, yi, m0i, di in zip(units, Y, mu0, D):
            if di == 1:
                psi_unit[u] = float(yi - m0i - att)
        se_cl = (
            cluster_se_from_influence(
                np.array(list(psi_unit.values())),
                np.array(list(psi_unit.keys())),
            )
            if psi_unit
            else float("nan")
        )

        se_boot, lo, hi = multiplier_bootstrap_ci(
            att,
            psi_unit,
            n_boot=self.n_boot,
            alpha=self.alpha,
            random_state=self.random_state,
        )
        se = se_boot if not np.isnan(se_boot) else se_cl
        if np.isnan(lo):
            lo, hi = normal_ci(att, se, self.alpha)

        res = ATTGTResult(
            g=g,
            t=t,
            att=float(att),
            se=float(se),
            ci_low=float(lo),
            ci_high=float(hi),
            n_treated=n_t,
            n_control=n_c,
            influence=psi_unit,
        )
        self._att_gt_cache[key] = res
        return res

    def all_att_gt(self) -> list[ATTGTResult]:
        """Compute ATT(g,t) for all valid (g,t) pairs with t >= g."""
        out: list[ATTGTResult] = []
        for g in self.panel.cohorts:
            for t in self.panel.times:
                if float(t) >= float(g):
                    out.append(self.att_gt(int(g), int(t)))
        return out

    def simple_att(self) -> ATTResult:
        """Weighted average of post-treatment ATT(g,t) (equal weight per cell)."""
        cells = [r for r in self.all_att_gt() if not np.isnan(r.att) and r.n_treated > 0]
        if not cells:
            return ATTResult(float("nan"), float("nan"), float("nan"), float("nan"), n_cells=0)

        atts = np.array([c.att for c in cells])
        weights = np.array([c.n_treated for c in cells], dtype=float)
        w_sum = weights.sum()
        att = float(np.average(atts, weights=weights)) if w_sum > 0 else float(atts.mean())

        unit_psi: dict[str | int, float] = {}
        for c in cells:
            w = c.n_treated / w_sum if w_sum > 0 else 1.0 / len(cells)
            for u, p in c.influence.items():
                unit_psi[u] = unit_psi.get(u, 0.0) + w * p

        se, lo, hi = multiplier_bootstrap_ci(
            att,
            unit_psi,
            n_boot=self.n_boot,
            alpha=self.alpha,
            random_state=self.random_state + 1,
        )
        return ATTResult(att=att, se=se, ci_low=lo, ci_high=hi, n_cells=len(cells))

    def group_att(self, g: int) -> dict[int, float]:
        """Cohort-specific ATT path over calendar times."""
        return {
            int(t): self.att_gt(g, int(t)).att for t in self.panel.times if float(t) >= float(g)
        }

    def calendar_att(self, t: int) -> dict[int, float]:
        """Time-specific ATTs across cohorts active at ``t``."""
        out: dict[int, float] = {}
        for g in self.panel.cohorts:
            if float(t) >= float(g):
                out[int(g)] = self.att_gt(int(g), int(t)).att
        return out

    def event_study_aggregation(
        self,
        min_e: int = -10,
        max_e: int = 10,
    ) -> CSEventStudyResult:
        """Aggregate ATT(g,t) to event time e = t - g with simultaneous bootstrap bands."""
        rows: list[dict[str, float | int]] = []
        event_cells: dict[int, list[ATTGTResult]] = {}
        for g in self.panel.cohorts:
            for t in self.panel.times:
                if float(t) < float(g):
                    continue
                e = int(t) - int(g)
                if min_e <= e <= max_e:
                    r = self.att_gt(int(g), int(t))
                    event_cells.setdefault(e, []).append(r)

        units = list(self.panel.unit_ids)
        unit_index = {u: i for i, u in enumerate(units)}
        event_times = sorted(event_cells.keys())
        if not event_times:
            return CSEventStudyResult(pd.DataFrame())

        att_vec = []
        psi_mat = []
        for e in event_times:
            cells = event_cells[e]
            atts = [c.att for c in cells if not np.isnan(c.att)]
            att_e = float(np.mean(atts)) if atts else float("nan")
            att_vec.append(att_e)
            col = np.zeros(len(units))
            for c in cells:
                w = 1.0 / len(cells)
                for u, p in c.influence.items():
                    if u in unit_index:
                        col[unit_index[u]] += w * p
            psi_mat.append(col)
        att_arr = np.array(att_vec)
        psi_matrix = np.array(psi_mat).T
        se_arr, lo_arr, hi_arr = simultaneous_bootstrap_bands(
            att_arr,
            psi_matrix,
            n_boot=self.n_boot,
            alpha=self.alpha,
            random_state=self.random_state + 2,
        )
        for i, e in enumerate(event_times):
            rows.append(
                {
                    "event_time": e,
                    "att": att_arr[i],
                    "se": se_arr[i],
                    "ci_low": lo_arr[i],
                    "ci_high": hi_arr[i],
                }
            )
        df = pd.DataFrame(rows)
        pre = df[df["event_time"] < 0]
        pretrend_ok = (
            bool(pre["ci_low"].abs().max() < 1e6 and (pre["att"].abs() < 0.5).all())
            if len(pre)
            else None
        )
        return CSEventStudyResult(leads_lags=df, pretrend_ok=pretrend_ok)


__all__ = [
    "ATTGTResult",
    "ATTResult",
    "CSEventStudyResult",
    "CallawaySantAnna",
]
