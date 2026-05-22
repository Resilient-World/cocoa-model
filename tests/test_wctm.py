"""Tests for Weighted Conformal Test Martingales."""

from __future__ import annotations

import math

from monitoring.conformal_cusum import ConformalCUSUM
from monitoring.wctm import (
    WeightedConformalTestMartingale,
    betting_function,
    weighted_conformal_pvalue,
)


def test_betting_function_integrates_to_one_per_epsilon() -> None:
    for eps in (-1, 0, 1):
        total = sum(betting_function(p, eps) for p in [i / 100 for i in range(1, 100)])
        assert 0.8 < total / 99 < 1.2


def test_martingale_detects_after_shift() -> None:
    wctm = WeightedConformalTestMartingale(alpha_fpr=0.01)
    for _ in range(300):
        wctm.update(5.0)
    assert wctm.detect() is not None
    assert wctm.log_martingale > wctm.log_threshold


def test_detect_threshold() -> None:
    wctm = WeightedConformalTestMartingale(alpha_fpr=0.01)
    for _ in range(500):
        wctm.update(4.0)
    alarm = wctm.detect()
    assert alarm is not None
    assert alarm.log_martingale > math.log(1.0 / 0.01)


def test_diagnose_concept_vs_covariate() -> None:
    wctm = WeightedConformalTestMartingale(alpha_fpr=0.01)
    wctm._log_martingale = 10.0
    wctm.set_x_alarm_active(False)
    assert wctm.diagnose() == "concept_shift"
    wctm.set_x_alarm_active(True)
    assert wctm.diagnose() == "covariate_shift"


def test_out_of_support() -> None:
    wctm = WeightedConformalTestMartingale(alpha_fpr=0.01, score_cap=5.0)
    wctm.update(10.0)
    alarm = wctm.detect()
    assert alarm is not None
    assert alarm.diagnosis == "out_of_support"


def test_weighted_pvalue_bounds() -> None:
    p = weighted_conformal_pvalue(2.0, [0.5, 1.0, 1.5], [1.0, 1.0, 1.0])
    assert 0.0 < p <= 1.0


def test_cusum_detects_shift() -> None:
    cusum = ConformalCUSUM(h=3.0, burn_in=10)
    for _ in range(100):
        cusum.update(0.0)
    assert not cusum.detect()
    for _ in range(50):
        cusum.update(3.0)
    assert cusum.detect()
