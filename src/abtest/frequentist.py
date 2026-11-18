"""Frequentist two-sample tests: two-proportion z-test, Welch t-test and Mann-Whitney U.

Every test has two layers:

* an array-level core (``*_pvalues`` / ``_*_core``) that broadcasts over many experiments at once
  and is used by the Monte-Carlo validation, and
* a scalar wrapper that returns a :class:`Comparison` with the effect, standard errors, a
  confidence interval and the relative lift.

The convention throughout the package is ``effect = treatment - control``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

Floats = NDArray[np.float64]


def z_quantile(alpha: float) -> float:
    """Two-sided normal critical value, e.g. 1.96 for ``alpha=0.05``."""
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


@dataclass(frozen=True)
class Comparison:
    """Treatment-minus-control estimate with a two-sided test.

    ``se`` is the unpooled standard error of the difference used for the confidence interval.
    The test statistic may use a different (pooled) standard error, as the z-test does.
    """

    method: str
    control_mean: float
    treatment_mean: float
    control_se: float
    treatment_se: float
    statistic: float
    p_value: float
    alpha: float = 0.05
    df: float | None = None

    @property
    def effect(self) -> float:
        return self.treatment_mean - self.control_mean

    @property
    def se(self) -> float:
        return math.hypot(self.control_se, self.treatment_se)

    @property
    def critical_value(self) -> float:
        if self.df is not None and math.isfinite(self.df):
            return float(stats.t.ppf(1.0 - self.alpha / 2.0, self.df))
        return z_quantile(self.alpha)

    @property
    def ci(self) -> tuple[float, float]:
        half = self.critical_value * self.se
        return self.effect - half, self.effect + half

    @property
    def relative_effect(self) -> float:
        if self.control_mean == 0:
            return math.nan
        return self.effect / self.control_mean

    @property
    def relative_se(self) -> float:
        """Delta-method standard error of ``treatment_mean / control_mean - 1``."""
        mc, mt = self.control_mean, self.treatment_mean
        if mc == 0:
            return math.nan
        return math.sqrt(self.treatment_se**2 / mc**2 + mt**2 * self.control_se**2 / mc**4)

    @property
    def relative_ci(self) -> tuple[float, float]:
        half = z_quantile(self.alpha) * self.relative_se
        return self.relative_effect - half, self.relative_effect + half

    def significant(self) -> bool:
        return self.p_value < self.alpha


@dataclass(frozen=True)
class RankComparison:
    """Mann-Whitney U result. ``prob_superiority`` = P(T > C) + 0.5 P(T = C)."""

    u_statistic: float
    z: float
    p_value: float
    prob_superiority: float
    method: str = "Mann-Whitney U (normal approx., tie + continuity corrected)"


# --------------------------------------------------------------------------- two proportions


def _ztest_core(
    x_c: ArrayLike, n_c: ArrayLike, x_t: ArrayLike, n_t: ArrayLike
) -> tuple[Floats, Floats]:
    x_c, n_c, x_t, n_t = (np.asarray(v, dtype=float) for v in (x_c, n_c, x_t, n_t))
    pooled = (x_c + x_t) / (n_c + n_t)
    se0 = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_c + 1.0 / n_t))
    diff = x_t / n_t - x_c / n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se0 > 0, diff / np.where(se0 > 0, se0, 1.0), 0.0)
    return z, 2.0 * stats.norm.sf(np.abs(z))


def ztest_pvalues(x_c: ArrayLike, n_c: ArrayLike, x_t: ArrayLike, n_t: ArrayLike) -> Floats:
    """Vectorised two-sided p-values of the pooled two-proportion z-test."""
    return _ztest_core(x_c, n_c, x_t, n_t)[1]


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> Comparison:
    """Pooled-variance z-test for H0: p_t = p_c, with an unpooled (Wald) confidence interval.

    The squared statistic equals Pearson's chi-square on the 2x2 table without Yates' correction.
    """
    if min(n_c, n_t) <= 0:
        raise ValueError("both groups need at least one observation")
    if not (0 <= x_c <= n_c and 0 <= x_t <= n_t):
        raise ValueError("successes must lie in [0, n]")
    z, p = _ztest_core(x_c, n_c, x_t, n_t)
    p_c, p_t = x_c / n_c, x_t / n_t
    return Comparison(
        method="two-proportion z-test",
        control_mean=p_c,
        treatment_mean=p_t,
        control_se=math.sqrt(p_c * (1 - p_c) / n_c),
        treatment_se=math.sqrt(p_t * (1 - p_t) / n_t),
        statistic=float(z),
        p_value=float(p),
        alpha=alpha,
    )


# --------------------------------------------------------------------------- Welch t-test


def _welch_core(
    mean_c: ArrayLike,
    var_c: ArrayLike,
    n_c: ArrayLike,
    mean_t: ArrayLike,
    var_t: ArrayLike,
    n_t: ArrayLike,
) -> tuple[Floats, Floats, Floats]:
    mean_c, var_c, n_c, mean_t, var_t, n_t = (
        np.asarray(v, dtype=float) for v in (mean_c, var_c, n_c, mean_t, var_t, n_t)
    )
    vc, vt = var_c / n_c, var_t / n_t
    se = np.sqrt(vc + vt)
    diff = mean_t - mean_c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, diff / np.where(se > 0, se, 1.0), 0.0)
        df = (vc + vt) ** 2 / (vc**2 / (n_c - 1) + vt**2 / (n_t - 1))
    df = np.where(np.isfinite(df), df, n_c + n_t - 2)
    return t, df, 2.0 * stats.t.sf(np.abs(t), df)


def welch_pvalues(control: ArrayLike, treatment: ArrayLike) -> Floats:
    """Vectorised Welch p-values along the last axis (one experiment per row)."""
    c, t = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    return _welch_core(
        c.mean(-1), c.var(-1, ddof=1), c.shape[-1], t.mean(-1), t.var(-1, ddof=1), t.shape[-1]
    )[2]


def welch_ttest_from_stats(
    mean_c: float,
    var_c: float,
    n_c: int,
    mean_t: float,
    var_t: float,
    n_t: int,
    alpha: float = 0.05,
    method: str = "Welch t-test",
) -> Comparison:
    """Welch's unequal-variance t-test from summary statistics (sample variances, ddof=1)."""
    if min(n_c, n_t) < 2:
        raise ValueError("Welch's test needs at least two observations per group")
    t, df, p = _welch_core(mean_c, var_c, n_c, mean_t, var_t, n_t)
    return Comparison(
        method=method,
        control_mean=float(mean_c),
        treatment_mean=float(mean_t),
        control_se=math.sqrt(var_c / n_c),
        treatment_se=math.sqrt(var_t / n_t),
        statistic=float(t),
        p_value=float(p),
        alpha=alpha,
        df=float(df),
    )


def welch_ttest(
    control: ArrayLike, treatment: ArrayLike, alpha: float = 0.05, method: str = "Welch t-test"
) -> Comparison:
    """Welch's t-test on raw observations."""
    c, t = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    return welch_ttest_from_stats(
        float(c.mean()),
        float(c.var(ddof=1)),
        c.size,
        float(t.mean()),
        float(t.var(ddof=1)),
        t.size,
        alpha=alpha,
        method=method,
    )


# --------------------------------------------------------------------------- Mann-Whitney U


def _tie_term(pooled: Floats) -> Floats:
    """Sum of (t^3 - t) over tie groups, per row of a 2-D array."""
    rows, width = pooled.shape
    s = np.sort(pooled, axis=-1)
    new_group = np.ones_like(s, dtype=bool)
    new_group[:, 1:] = s[:, 1:] != s[:, :-1]
    starts = np.flatnonzero(new_group.ravel())
    lengths = np.diff(np.append(starts, s.size)).astype(float)
    return np.bincount(starts // width, weights=lengths**3 - lengths, minlength=rows)


def _mann_whitney_core(control: Floats, treatment: Floats) -> tuple[Floats, Floats, Floats]:
    c = np.atleast_2d(np.asarray(control, dtype=float))
    t = np.atleast_2d(np.asarray(treatment, dtype=float))
    n_t, n_c = t.shape[-1], c.shape[-1]
    pooled = np.concatenate([t, c], axis=-1)
    ranks = stats.rankdata(pooled, axis=-1)
    u = ranks[:, :n_t].sum(axis=-1) - n_t * (n_t + 1) / 2.0
    n = n_t + n_c
    sigma = np.sqrt(n_t * n_c / 12.0 * ((n + 1) - _tie_term(pooled) / (n * (n - 1))))
    numerator = np.maximum(np.abs(u - n_t * n_c / 2.0) - 0.5, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, numerator / np.where(sigma > 0, sigma, 1.0), 0.0)
    z = np.sign(u - n_t * n_c / 2.0) * z
    return u, z, np.minimum(1.0, 2.0 * stats.norm.sf(np.abs(z)))


def mann_whitney_pvalues(control: ArrayLike, treatment: ArrayLike) -> Floats:
    """Vectorised two-sided Mann-Whitney p-values, one experiment per row."""
    return _mann_whitney_core(np.asarray(control), np.asarray(treatment))[2]


def mann_whitney(control: ArrayLike, treatment: ArrayLike) -> RankComparison:
    """Two-sided Mann-Whitney U test of treatment against control.

    It tests whether one group tends to produce larger values (stochastic ordering), not whether
    the means differ. On zero-inflated revenue it is driven largely by the conversion difference.
    """
    c, t = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    u, z, p = _mann_whitney_core(c, t)
    return RankComparison(
        u_statistic=float(u[0]),
        z=float(z[0]),
        p_value=float(p[0]),
        prob_superiority=float(u[0]) / (c.size * t.size),
    )


def normal_test(effect: float, se: float) -> tuple[float, float]:
    """z statistic and two-sided p-value for an asymptotically normal estimate."""
    if se <= 0:
        return 0.0, 1.0 if effect == 0 else 0.0
    z = effect / se
    return z, float(2.0 * stats.norm.sf(abs(z)))
