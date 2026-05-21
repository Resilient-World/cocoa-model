"""
Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).

Unit and time weights via ridge-regularized simplex QPs (cvxpy), ATT with
pre-period correction, jackknife SE (multiple treated units) or placebo SE
(single treated unit). Staggered adoption via block-by-block aggregation
(Appendix B).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from analysis._staggered_did_common import (
    normal_ci,
    prepare_staggered_panel,
)

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SyntheticDiD requires cvxpy>=1.5. Install with: pip install 'cvxpy>=1.5'"
    ) from exc


@dataclass
class SDIDResult:
    """Synthetic DiD ATT estimate."""

    att: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float | None
    omega_hat: np.ndarray
    lambda_hat: np.ndarray
    n_treated: int
    n_control: int
    method: str = "synthdid"
    pretrend_pvalue: float | None = None
    block_atts: list[float] = field(default_factory=list)


@dataclass
class SDIDSetup:
    """Balanced outcome matrix for one SDID block (controls first, treated last)."""

    Y: np.ndarray
    N0: int
    T0: int
    unit_ids: list[Any]
    times: list[Any]


def _noise_level(Y: np.ndarray, N0: int, T0: int) -> float:
    """R ``noise.level``: SD of first differences on control pre-period rows."""
    pre = Y[:N0, :T0]
    if pre.shape[0] < 2 or pre.shape[1] < 2:
        return float(np.std(Y))
    diffs = np.diff(pre, axis=1)
    row_diffs = np.diff(pre, axis=0)
    vals = np.concatenate([diffs.ravel(), row_diffs.ravel()])
    return float(np.std(vals)) if len(vals) else float(np.std(Y))


def _default_zeta_omega(N1: int, T1: int, noise: float) -> float:
    return float((N1 * T1) ** 0.25 * noise)


def _collapsed_form(Y: np.ndarray, N0: int, T0: int) -> np.ndarray:
    """Collapse treated rows/columns to means (R ``collapsed.form``)."""
    n, t = Y.shape
    top_left = Y[:N0, :T0]
    top_right = Y[:N0, T0:].mean(axis=1, keepdims=True) if T0 < t else np.empty((N0, 0))
    bottom_left = Y[N0:, :T0].mean(axis=0, keepdims=True) if N0 < n else np.empty((0, T0))
    bottom_right = np.array([[Y[N0:, T0:].mean()]]) if (N0 < n and T0 < t) else np.empty((0, 0))
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def _fw_step(
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    eta: float,
) -> np.ndarray:
    """One Frank-Wolfe step (R ``fw.step``)."""
    Ax = A @ x
    half_grad = (Ax - b) @ A + eta * x
    i = int(np.argmin(half_grad))
    d_x = -x.copy()
    d_x[i] = 1.0 - x[i]
    if np.allclose(d_x, 0):
        return x
    d_err = A[:, i] - Ax
    step = -float(half_grad @ d_x) / (float(np.sum(d_err**2)) + eta * float(np.sum(d_x**2)))
    constrained = min(1.0, max(0.0, step))
    return x + constrained * d_x


def _sc_weight_fw(
    Y: np.ndarray,
    zeta: float,
    *,
    intercept: bool = True,
    max_iter: int = 1000,
    min_decrease: float = 1e-3,
) -> np.ndarray:
    """Frank-Wolfe synthetic-control weights (R ``sc.weight.fw``)."""
    T0 = Y.shape[1] - 1
    N0 = Y.shape[0]
    lam = np.full(T0, 1.0 / T0)
    if intercept:
        Y = Y - Y.mean(axis=0, keepdims=True)
    A = Y[:, :T0]
    b = Y[:, T0]
    eta = N0 * (zeta**2)
    prev_val = np.inf
    for t in range(max_iter):
        lam = _fw_step(A, lam, b, eta)
        err = Y[:N0, :] @ np.concatenate([lam, [-1.0]])
        val = (zeta**2) * float(np.sum(lam**2)) + float(np.sum(err**2)) / N0
        if t >= 2 and prev_val - val <= min_decrease**2:
            break
        prev_val = val
    s = lam.sum()
    return lam / s if s > 0 else np.ones(T0) / T0


def _solve_simplex_weights_cvxpy(
    A: np.ndarray,
    target: np.ndarray,
    *,
    zeta: float,
) -> np.ndarray:
    """cvxpy simplex QP fallback (ridge + sum-to-one)."""
    n = A.shape[1]
    w = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(A @ w - target) + zeta * cp.sum_squares(w))
    constraints = [cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except cp.SolverError:
        prob.solve(verbose=False)
    if w.value is None:
        raise ValueError("SDID weight optimization failed to converge")
    out = np.maximum(np.asarray(w.value, dtype=float).ravel(), 0.0)
    s = out.sum()
    return out / s if s > 0 else np.ones(n) / n


def _estimate_omega(
    Y: np.ndarray,
    N0: int,
    T0: int,
    *,
    zeta: float,
) -> np.ndarray:
    """Unit weights on control donors (length N0)."""
    Yc = _collapsed_form(Y, N0, T0)
    Y_fw = Yc[:, :T0].T  # T0 x (N0+1)
    return _sc_weight_fw(Y_fw, zeta, intercept=True)


def _estimate_lambda(
    Y: np.ndarray,
    N0: int,
    T0: int,
    *,
    zeta: float,
) -> np.ndarray:
    """Time weights on pre periods (length T0)."""
    Yc = _collapsed_form(Y, N0, T0)
    return _sc_weight_fw(Yc[:N0, :], zeta, intercept=True)


def synthdid_att(
    Y: np.ndarray,
    N0: int,
    T0: int,
    *,
    zeta_omega: float | None = None,
    zeta_lambda: float = 1e-6,
    omega: np.ndarray | None = None,
    lam: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Point ATT (R ``synthdid_estimate`` matrix formula).

    tau = [-omega; 1/N1]' Y [-lambda; 1/T1]
    """
    n, t = Y.shape
    N1 = n - N0
    T1 = t - T0
    if N1 < 1 or T1 < 1 or N0 < 1 or T0 < 1:
        raise ValueError(f"Invalid SDID dimensions N0={N0}, N1={N1}, T0={T0}, T1={T1}")

    noise = _noise_level(Y, N0, T0)
    if zeta_omega is None:
        zeta_omega = _default_zeta_omega(N1, T1, noise)

    if omega is None:
        omega = _estimate_omega(Y, N0, T0, zeta=zeta_omega)
    if lam is None:
        lam = _estimate_lambda(Y, N0, T0, zeta=zeta_lambda * noise)

    w_unit = np.concatenate([-omega, np.full(N1, 1.0 / N1)])
    w_time = np.concatenate([-lam, np.full(T1, 1.0 / T1)])
    tau = float(w_unit @ Y @ w_time)
    return tau, omega, lam


def _jackknife_se(
    Y: np.ndarray,
    N0: int,
    T0: int,
    *,
    zeta_omega: float | None,
    zeta_lambda: float,
    omega: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Fixed-weight jackknife over units (Algorithm 3)."""
    n = Y.shape[0]
    if n - N0 <= 1:
        return float("nan")
    u = np.empty(n)
    for i in range(n):
        ind_bool = np.ones(n, dtype=bool)
        ind_bool[i] = False
        Y_i = Y[ind_bool]
        if i < N0:
            n0_i = N0 - 1
            om_i = _sum_normalize(np.delete(omega, i))
        else:
            n0_i = N0
            om_i = _sum_normalize(omega)
        if n0_i < 1 or Y_i.shape[0] - n0_i < 1:
            u[i] = np.nan
            continue
        try:
            u[i], _, _ = synthdid_att(
                Y_i,
                n0_i,
                T0,
                zeta_omega=zeta_omega,
                zeta_lambda=zeta_lambda,
                omega=om_i,
                lam=lam,
            )
        except (ValueError, cp.SolverError):
            u[i] = np.nan
    valid = u[~np.isnan(u)]
    if len(valid) < 2:
        return float("nan")
    return float(np.sqrt(((n - 1) / n) * (n - 1) * np.var(valid, ddof=1)))


def _sum_normalize(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    if s > 0:
        return x / s
    return np.ones(len(x)) / len(x)


def _placebo_se(
    Y: np.ndarray,
    N0: int,
    T0: int,
    *,
    zeta_omega: float | None,
    zeta_lambda: float,
    omega: np.ndarray,
    lam: np.ndarray,
    n_replications: int = 200,
    random_state: int = 42,
) -> tuple[float, float | None]:
    """Placebo SE (Algorithm 4) — resample control units as pseudo-treated."""
    N1 = Y.shape[0] - N0
    if N0 <= N1:
        return float("nan"), None
    rng = np.random.default_rng(random_state)
    boots: list[float] = []
    for _ in range(n_replications * 5):
        if len(boots) >= n_replications:
            break
        ind = rng.choice(N0, size=N0, replace=True)
        n0_boot = len(ind) - N1
        if n0_boot < 1:
            continue
        Y_boot = Y[ind]
        om_boot = _sum_normalize(omega[ind[:n0_boot]])
        try:
            tau, _, _ = synthdid_att(
                Y_boot,
                n0_boot,
                T0,
                zeta_omega=zeta_omega,
                zeta_lambda=zeta_lambda,
                omega=om_boot,
                lam=lam,
            )
            boots.append(tau)
        except (ValueError, cp.SolverError):
            continue
    if len(boots) < 10:
        return float("nan"), None
    boots_arr = np.array(boots)
    se = float(np.sqrt((len(boots) - 1) / len(boots) * boots_arr.var(ddof=1)))
    return se, None


def panel_to_sdid_setup(
    panel: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit_mask: np.ndarray | pd.Series,
    control_unit_mask: np.ndarray | pd.Series,
    treatment_start: float,
) -> SDIDSetup:
    """Build balanced Y matrix (controls first, treated last) for simultaneous block at ``treatment_start``."""
    sub = panel.copy()
    tu = sub.loc[treated_unit_mask, unit_col].unique()
    cu = sub.loc[control_unit_mask, unit_col].unique()
    if len(tu) < 1 or len(cu) < 1:
        raise ValueError("SDID block requires at least one treated and one control unit")

    times = np.sort(sub[time_col].unique())
    t0_idx = int(np.searchsorted(times, treatment_start))
    if t0_idx < 1 or t0_idx >= len(times):
        raise ValueError(f"Invalid treatment_start={treatment_start} for times {times}")

    rows: list[np.ndarray] = []
    unit_ids: list[Any] = []
    for uid in list(cu) + list(tu):
        udf = sub.loc[sub[unit_col] == uid].sort_values(time_col)
        if len(udf) != len(times):
            raise ValueError(f"Unbalanced panel for unit {uid}")
        rows.append(udf[outcome_col].to_numpy(dtype=float))
        unit_ids.append(uid)

    Y = np.vstack(rows)
    N0 = len(cu)
    return SDIDSetup(Y=Y, N0=N0, T0=t0_idx, unit_ids=unit_ids, times=list(times))


class SyntheticDiD:
    """
    Synthetic DiD for staggered panels (Arkhangelsky et al. 2021).

    Single-cohort panels map to one SDID block; multiple cohorts use Appendix B
    block aggregation with not-yet-treated controls per block.
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,
        *,
        unit_col: str = "farm_id",
        time_col: str = "period",
        treat_time_col: str = "treatment_period",
        outcome_col: str = "yield",
        zeta_omega: float | None = None,
        zeta_lambda: float = 1e-6,
        n_placebo: int = 200,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.panel = prepare_staggered_panel(
            panel_df,
            unit_col=unit_col,
            time_col=time_col,
            treat_time_col=treat_time_col,
            outcome_col=outcome_col,
        )
        self.zeta_omega = zeta_omega
        self.zeta_lambda = zeta_lambda
        self.n_placebo = n_placebo
        self.alpha = alpha
        self.random_state = random_state

    def _block_masks(self, g: float) -> tuple[np.ndarray, np.ndarray]:
        """Treated cohort g vs not-yet-treated + never-treated controls at adoption."""
        df = self.panel.df
        g_arr = df["_G"].to_numpy(dtype=float)
        unit = df[self.panel.unit_col]
        treated_units = df.loc[g_arr == float(g), self.panel.unit_col].unique()
        never = np.isnan(g_arr)
        not_yet = (~np.isnan(g_arr)) & (g_arr > float(g))
        control_units = df.loc[never | not_yet, self.panel.unit_col].unique()
        tu_mask = unit.isin(treated_units).to_numpy()
        cu_mask = unit.isin(control_units).to_numpy()
        return tu_mask, cu_mask

    def _estimate_block(self, g: float) -> tuple[float, np.ndarray, np.ndarray, SDIDSetup]:
        tu_mask, cu_mask = self._block_masks(g)
        setup = panel_to_sdid_setup(
            self.panel.df,
            unit_col=self.panel.unit_col,
            time_col=self.panel.time_col,
            outcome_col=self.panel.outcome_col,
            treated_unit_mask=tu_mask,
            control_unit_mask=cu_mask,
            treatment_start=float(g),
        )
        tau, omega, lam = synthdid_att(
            setup.Y,
            setup.N0,
            setup.T0,
            zeta_omega=self.zeta_omega,
            zeta_lambda=self.zeta_lambda,
        )
        return tau, omega, lam, setup

    def estimate(self) -> SDIDResult:
        """Run SDID (possibly multi-block) and return ATT with inference."""
        cohorts = self.panel.cohorts
        if len(cohorts) == 0:
            raise ValueError("No treated cohorts in panel")

        block_results: list[tuple[float, int, np.ndarray, np.ndarray, SDIDSetup]] = []
        for g in cohorts:
            try:
                tau, omega, lam, setup = self._estimate_block(float(g))
            except (ValueError, cp.SolverError):
                continue
            n_tr = setup.Y.shape[0] - setup.N0
            block_results.append((tau, n_tr, omega, lam, setup))

        if not block_results:
            raise ValueError("SDID failed for all cohort blocks")

        weights = np.array([r[1] for r in block_results], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(block_results))
        weights /= weights.sum()
        att = float(sum(w * r[0] for w, r in zip(weights, block_results)))
        block_atts = [r[0] for r in block_results]

        # Pooled inference on largest block (most treated units)
        main = max(block_results, key=lambda r: r[1])
        tau_m, n_tr, omega, lam, setup = main
        noise = _noise_level(setup.Y, setup.N0, setup.T0)
        z_om = self.zeta_omega or _default_zeta_omega(
            setup.Y.shape[0] - setup.N0, setup.Y.shape[1] - setup.T0, noise
        )

        if n_tr > 1:
            se = _jackknife_se(
                setup.Y,
                setup.N0,
                setup.T0,
                zeta_omega=z_om,
                zeta_lambda=self.zeta_lambda,
                omega=omega,
                lam=lam,
            )
        else:
            se, _ = _placebo_se(
                setup.Y,
                setup.N0,
                setup.T0,
                zeta_omega=z_om,
                zeta_lambda=self.zeta_lambda,
                omega=omega,
                lam=lam,
                n_replications=self.n_placebo,
                random_state=self.random_state,
            )

        if np.isnan(se) or se <= 0:
            se = float("nan")

        lo, hi = normal_ci(att, se, self.alpha)
        p_value = None
        if se is not None and not np.isnan(se) and se > 0:
            from math import erf, sqrt

            z = att / se
            p_value = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))))

        n_tr_total = int(sum(r[1] for r in block_results))
        n_co = int(setup.N0)

        return SDIDResult(
            att=att,
            se=se,
            ci_low=lo,
            ci_high=hi,
            p_value=p_value,
            omega_hat=omega,
            lambda_hat=lam,
            n_treated=n_tr_total,
            n_control=n_co,
            block_atts=block_atts,
        )

    def pretrend_placebo_pvalue(self) -> float | None:
        """Placebo estimate on pre-treatment periods only (R ``synthdid_placebo``)."""
        cohorts = self.panel.cohorts
        if len(cohorts) == 0:
            return None
        g = float(cohorts[0])
        tu_mask, cu_mask = self._block_masks(g)
        try:
            setup = panel_to_sdid_setup(
                self.panel.df,
                unit_col=self.panel.unit_col,
                time_col=self.panel.time_col,
                outcome_col=self.panel.outcome_col,
                treated_unit_mask=tu_mask,
                control_unit_mask=cu_mask,
                treatment_start=g,
            )
        except ValueError:
            return None
        T0 = setup.T0
        if T0 < 3:
            return None
        frac = 1.0 - T0 / setup.Y.shape[1]
        placebo_T0 = max(1, int(np.floor(T0 * (1.0 - frac))))
        Y_pre = setup.Y[:, :T0]
        try:
            tau, _, _ = synthdid_att(
                Y_pre,
                setup.N0,
                placebo_T0,
                zeta_omega=self.zeta_omega,
                zeta_lambda=self.zeta_lambda,
            )
        except (ValueError, cp.SolverError):
            return None
        se, _ = _placebo_se(
            Y_pre,
            setup.N0,
            placebo_T0,
            zeta_omega=self.zeta_omega,
            zeta_lambda=self.zeta_lambda,
            omega=_estimate_omega(Y_pre, setup.N0, placebo_T0, zeta=self.zeta_omega or 1e-3),
            lam=_estimate_lambda(Y_pre, setup.N0, placebo_T0, zeta=self.zeta_lambda),
            n_replications=min(50, self.n_placebo),
            random_state=self.random_state,
        )
        if se is None or np.isnan(se) or se <= 0:
            return None
        from math import erf, sqrt

        z = abs(tau / se)
        return float(2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))


__all__ = ["SDIDResult", "SDIDSetup", "SyntheticDiD", "panel_to_sdid_setup", "synthdid_att"]
