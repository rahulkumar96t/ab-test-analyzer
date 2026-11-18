"""Bayesian summaries: beta-binomial for conversion, a normal model for continuous metrics.

Both report the quantities a product decision needs:

* ``prob_better``: posterior probability that treatment beats control;
* ``expected_loss_ship``: E[max(control - treatment, 0)], how much we expect to lose (in metric
  units) if we ship treatment and it is in fact worse;
* ``expected_loss_keep``: the same for keeping control.

A common rule is to ship when the expected loss of shipping falls below a tolerance. Neither
summary is a protection against peeking: a Bayesian rule checked every day also stops on noise
more often than checked once (see the README).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import integrate, stats

from abtest.frequentist import z_quantile


@dataclass(frozen=True)
class BayesResult:
    model: str
    prob_better: float
    expected_loss_ship: float
    expected_loss_keep: float
    diff_mean: float
    ci_low: float
    ci_high: float

    def lower_is_better(self) -> BayesResult:
        """Re-express the summary for a metric where a decrease is the improvement."""
        return replace(
            self,
            prob_better=1.0 - self.prob_better,
            expected_loss_ship=self.expected_loss_keep,
            expected_loss_keep=self.expected_loss_ship,
        )


def _integrate_over(dist: Any, func: Callable[[float], float]) -> float:
    lo, hi = float(dist.ppf(1e-12)), float(dist.isf(1e-12))
    value, _ = integrate.quad(func, lo, hi, limit=200, epsabs=1e-13, epsrel=1e-10)
    return float(value)


def prob_beta_greater(a_t: float, b_t: float, a_c: float, b_c: float) -> float:
    """P(T > C) for independent T ~ Beta(a_t, b_t), C ~ Beta(a_c, b_c), by quadrature."""
    post_t, post_c = stats.beta(a_t, b_t), stats.beta(a_c, b_c)
    return min(1.0, max(0.0, _integrate_over(post_t, lambda x: post_t.pdf(x) * post_c.cdf(x))))


def expected_shortfall_beta(a_t: float, b_t: float, a_c: float, b_c: float) -> float:
    """E[max(C - T, 0)] for independent beta variables.

    Uses E[(C - y)^+] = mean_C * P(C' > y) - y * P(C > y) with C' ~ Beta(a_c + 1, b_c), which
    follows from x * pdf_C(x) = mean_C * pdf_C'(x); the outer expectation over T is a quadrature.
    """
    post_t, post_c, shifted = stats.beta(a_t, b_t), stats.beta(a_c, b_c), stats.beta(a_c + 1, b_c)
    mean_c = a_c / (a_c + b_c)

    def inner(y: float) -> float:
        return post_t.pdf(y) * (mean_c * shifted.sf(y) - y * post_c.sf(y))

    return max(0.0, _integrate_over(post_t, inner))


def beta_binomial(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    prior: tuple[float, float] = (1.0, 1.0),
    credible: float = 0.95,
    draws: int = 200_000,
    seed: int | None = 0,
) -> BayesResult:
    """Independent Beta(prior) - binomial models for the two conversion rates."""
    a0, b0 = prior
    a_c, b_c = a0 + x_c, b0 + n_c - x_c
    a_t, b_t = a0 + x_t, b0 + n_t - x_t
    loss_ship = expected_shortfall_beta(a_t, b_t, a_c, b_c)
    diff_mean = a_t / (a_t + b_t) - a_c / (a_c + b_c)
    rng = np.random.default_rng(seed)
    diff = rng.beta(a_t, b_t, draws) - rng.beta(a_c, b_c, draws)
    tail = (1.0 - credible) / 2.0
    lo, hi = np.quantile(diff, [tail, 1.0 - tail])
    return BayesResult(
        model=f"beta-binomial, Beta({a0:g}, {b0:g}) prior",
        prob_better=prob_beta_greater(a_t, b_t, a_c, b_c),
        expected_loss_ship=loss_ship,
        expected_loss_keep=loss_ship + diff_mean,
        diff_mean=diff_mean,
        ci_low=float(lo),
        ci_high=float(hi),
    )


def normal_posterior(
    effect: float,
    se: float,
    prior_mean: float = 0.0,
    prior_sd: float | None = None,
    credible: float = 0.95,
) -> BayesResult:
    """Normal likelihood for the estimated difference with a flat or normal prior on the effect.

    With large samples the difference of means is approximately normal with the estimated
    standard error, so the posterior of the true difference is normal as well.
    """
    if se <= 0:
        raise ValueError("standard error must be positive")
    if prior_sd is None:
        mean, sd, label = effect, se, "normal model, flat prior"
    else:
        precision = 1.0 / se**2 + 1.0 / prior_sd**2
        mean = (effect / se**2 + prior_mean / prior_sd**2) / precision
        sd = math.sqrt(1.0 / precision)
        label = f"normal model, N({prior_mean:g}, {prior_sd:g}^2) prior"
    ratio = mean / sd
    pdf, cdf = float(stats.norm.pdf(ratio)), float(stats.norm.cdf(ratio))
    half = z_quantile(1.0 - credible) * sd
    return BayesResult(
        model=label,
        prob_better=cdf,
        expected_loss_ship=sd * pdf - mean * (1.0 - cdf),
        expected_loss_keep=sd * pdf + mean * cdf,
        diff_mean=mean,
        ci_low=mean - half,
        ci_high=mean + half,
    )
