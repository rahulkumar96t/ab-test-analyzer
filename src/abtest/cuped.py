"""CUPED: variance reduction with a pre-experiment covariate (Deng et al., 2013).

For a metric Y and a covariate X measured before assignment (for example last month's revenue
of the same user), the adjusted metric is

    Y_cv = Y - theta * (X - mean(X)),   theta = cov(X, Y) / var(X).

Because X is fixed before randomisation its expectation is the same in both arms, so the
difference of adjusted means estimates the same treatment effect as the raw difference. Its
variance is smaller by a factor (1 - rho^2), where rho is the correlation of X and Y.

``theta`` is estimated once on the pooled data of both arms. Estimating it per arm would let the
treatment effect leak into the adjustment; using the pooled estimate keeps the estimator
consistent (the remaining bias is of order 1/n, see ``tests/test_cuped.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from abtest.frequentist import Comparison, welch_ttest


@dataclass(frozen=True)
class CupedResult:
    raw: Comparison
    adjusted: Comparison
    theta: float
    correlation: float

    @property
    def variance_reduction(self) -> float:
        """Measured reduction of the variance of the effect estimate, 1 - (se_adj / se_raw)^2."""
        return 1.0 - (self.adjusted.se / self.raw.se) ** 2


def cuped_theta(y: ArrayLike, x: ArrayLike) -> float:
    """OLS slope of y on x; 0 when the covariate is constant."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    var_x = x.var(ddof=1)
    if var_x == 0:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def cuped_adjust(
    y: ArrayLike, x: ArrayLike, theta: float, x_mean: float | None = None
) -> NDArray[np.float64]:
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    center = float(x.mean()) if x_mean is None else x_mean
    return y - theta * (x - center)


def cuped_test(
    y_control: ArrayLike,
    x_control: ArrayLike,
    y_treatment: ArrayLike,
    x_treatment: ArrayLike,
    alpha: float = 0.05,
) -> CupedResult:
    """Welch test on raw and CUPED-adjusted values, theta and centring from the pooled sample."""
    yc, xc = np.asarray(y_control, dtype=float), np.asarray(x_control, dtype=float)
    yt, xt = np.asarray(y_treatment, dtype=float), np.asarray(x_treatment, dtype=float)
    if yc.shape != xc.shape or yt.shape != xt.shape:
        raise ValueError("metric and covariate must have the same length in each arm")
    y_all, x_all = np.concatenate([yc, yt]), np.concatenate([xc, xt])
    theta = cuped_theta(y_all, x_all)
    x_mean = float(x_all.mean())
    std_x, std_y = x_all.std(), y_all.std()
    corr = float(np.corrcoef(x_all, y_all)[0, 1]) if std_x > 0 and std_y > 0 else 0.0
    raw = welch_ttest(yc, yt, alpha=alpha)
    adjusted = welch_ttest(
        cuped_adjust(yc, xc, theta, x_mean),
        cuped_adjust(yt, xt, theta, x_mean),
        alpha=alpha,
        method="Welch t-test on CUPED-adjusted values",
    )
    return CupedResult(raw=raw, adjusted=adjusted, theta=theta, correlation=corr)
