"""Ratio metrics with the delta method.

Revenue per session, clicks per page view and similar metrics are ratios of two sums, while the
experiment randomises users. Sessions of the same user are correlated, so treating every
session as an independent observation underestimates the variance and inflates false
positives (measured in ``abtest validate``). The delta method works at the randomisation unit:
for per-user numerator X and denominator Y,

    Var(mean(X) / mean(Y)) ~ (1/n) * (s_xx / my^2 - 2 mx s_xy / my^3 + mx^2 s_yy / my^4)

(Deng, Knoblich and Lu, 2018, "Applying the Delta Method in Metric Analytics").
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from abtest.frequentist import Comparison, normal_test

Floats = NDArray[np.float64]


def ratio_moments(numerator: ArrayLike, denominator: ArrayLike) -> tuple[Floats, Floats]:
    """Ratio of sums and its delta-method variance along the last axis (vectorised)."""
    x, y = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    n = x.shape[-1]
    mx, my = x.mean(axis=-1), y.mean(axis=-1)
    dx, dy = x - mx[..., None], y - my[..., None]
    sxx = (dx * dx).sum(axis=-1) / (n - 1)
    syy = (dy * dy).sum(axis=-1) / (n - 1)
    sxy = (dx * dy).sum(axis=-1) / (n - 1)
    var = (sxx / my**2 - 2 * mx * sxy / my**3 + mx**2 * syy / my**4) / n
    return mx / my, np.maximum(var, 0.0)


def delta_ratio_pvalues(
    num_control: ArrayLike,
    den_control: ArrayLike,
    num_treatment: ArrayLike,
    den_treatment: ArrayLike,
) -> Floats:
    """Vectorised two-sided p-values of the delta-method test, one experiment per row."""
    r_c, v_c = ratio_moments(num_control, den_control)
    r_t, v_t = ratio_moments(num_treatment, den_treatment)
    return 2.0 * stats.norm.sf(np.abs(r_t - r_c) / np.sqrt(v_c + v_t))


def ratio_estimate(numerator: ArrayLike, denominator: ArrayLike) -> tuple[float, float]:
    """Ratio of sums and its delta-method standard error from per-unit totals."""
    x, y = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("numerator and denominator must be 1-D arrays of equal length >= 2")
    if y.sum() == 0:
        raise ValueError("denominator sums to zero")
    ratio, var = ratio_moments(x, y)
    return float(ratio), math.sqrt(float(var))


def delta_ratio_test(
    num_control: ArrayLike,
    den_control: ArrayLike,
    num_treatment: ArrayLike,
    den_treatment: ArrayLike,
    alpha: float = 0.05,
) -> Comparison:
    """z-test of the difference of two ratio metrics with delta-method standard errors."""
    r_c, se_c = ratio_estimate(num_control, den_control)
    r_t, se_t = ratio_estimate(num_treatment, den_treatment)
    z, p = normal_test(r_t - r_c, math.hypot(se_c, se_t))
    return Comparison(
        method="delta-method z-test (ratio of sums)",
        control_mean=r_c,
        treatment_mean=r_t,
        control_se=se_c,
        treatment_se=se_t,
        statistic=z,
        p_value=p,
        alpha=alpha,
    )
