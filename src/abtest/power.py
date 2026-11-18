"""Sample size, power and minimum detectable effect (MDE) for two-arm tests.

* Proportions: the pooled z-test formula without continuity correction (Fleiss, Levin and Paik,
  "Statistical Methods for Rates and Proportions"), with allocation ratio k.
* Means: the normal approximation, optionally refined with the noncentral t distribution, which
  reproduces Cohen's tables (64 per group for d = 0.5, alpha 0.05, power 0.8).

``ratio`` is the treatment / control allocation ratio, so ``ratio=1`` is a 50/50 split. CUPED
enters through ``variance_reduction``: a covariate with correlation rho removes rho^2 of the
variance, which shrinks the required sample by the same factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize, stats

from abtest.frequentist import z_quantile


@dataclass(frozen=True)
class SampleSize:
    control: int
    treatment: int

    @property
    def total(self) -> int:
        return self.control + self.treatment


def _split(n_control: float, ratio: float) -> SampleSize:
    n_c = math.ceil(n_control - 1e-9)
    return SampleSize(control=n_c, treatment=math.ceil(n_c * ratio - 1e-9))


def _treatment_rate(baseline: float, mde: float, relative: bool) -> float:
    p2 = baseline * (1.0 + mde) if relative else baseline + mde
    if not (0.0 < baseline < 1.0 and 0.0 < p2 < 1.0) or p2 == baseline:
        raise ValueError("baseline and baseline + effect must be distinct rates in (0, 1)")
    return p2


# ------------------------------------------------------------------------------ proportions


def sample_size_proportions(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    relative: bool = False,
    ratio: float = 1.0,
) -> SampleSize:
    """Users per arm for a two-sided pooled z-test to detect ``baseline -> baseline + mde``."""
    p1 = baseline
    p2 = _treatment_rate(baseline, mde, relative)
    p_bar = (p1 + ratio * p2) / (1.0 + ratio)
    null_sd = math.sqrt(p_bar * (1 - p_bar) * (1 + 1 / ratio))
    alt_sd = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    z_a, z_b = z_quantile(alpha), float(stats.norm.ppf(power))
    return _split((z_a * null_sd + z_b * alt_sd) ** 2 / (p2 - p1) ** 2, ratio)


def power_proportions(
    baseline: float,
    mde: float,
    n_control: int,
    alpha: float = 0.05,
    relative: bool = False,
    ratio: float = 1.0,
) -> float:
    """Approximate power of the pooled z-test (both rejection tails)."""
    p1 = baseline
    p2 = _treatment_rate(baseline, mde, relative)
    n_t = n_control * ratio
    p_bar = (p1 * n_control + p2 * n_t) / (n_control + n_t)
    null_se = math.sqrt(p_bar * (1 - p_bar) * (1 / n_control + 1 / n_t))
    alt_se = math.sqrt(p1 * (1 - p1) / n_control + p2 * (1 - p2) / n_t)
    z_a, delta = z_quantile(alpha), abs(p2 - p1)
    upper = stats.norm.sf((z_a * null_se - delta) / alt_se)
    lower = stats.norm.cdf((-z_a * null_se - delta) / alt_se)
    return float(upper + lower)


def mde_proportions(
    baseline: float,
    n_control: int,
    alpha: float = 0.05,
    power: float = 0.8,
    relative: bool = False,
    ratio: float = 1.0,
) -> float:
    """Smallest upward effect detectable with the given power (absolute or relative)."""
    upper = (1.0 - baseline) * (1 - 1e-9)

    def gap(delta: float) -> float:
        return power_proportions(baseline, delta, n_control, alpha, False, ratio) - power

    if gap(upper) < 0:
        raise ValueError("no effect is detectable with this sample size and power")
    delta = optimize.brentq(gap, 1e-12, upper, xtol=1e-12)
    return delta / baseline if relative else delta


# ------------------------------------------------------------------------------------ means


def power_means(
    mde: float,
    sd: float,
    n_control: int,
    alpha: float = 0.05,
    ratio: float = 1.0,
    variance_reduction: float = 0.0,
    t_distribution: bool = False,
    sd_treatment: float | None = None,
) -> float:
    """Power of a two-sided two-sample test of means.

    ``sd`` is the control standard deviation; ``sd_treatment`` defaults to the same value. A
    treatment that scales the metric usually scales its spread too, and planning with the
    control SD alone then overstates the power (see the validation results in the README).
    """
    n_t = n_control * ratio
    sd_t = sd if sd_treatment is None else sd_treatment
    se = math.sqrt((1.0 - variance_reduction) * (sd**2 / n_control + sd_t**2 / n_t))
    shift = abs(mde) / se
    if not t_distribution:
        z_a = z_quantile(alpha)
        return float(stats.norm.sf(z_a - shift) + stats.norm.cdf(-z_a - shift))
    df = n_control + n_t - 2
    t_a = float(stats.t.ppf(1 - alpha / 2, df))
    return float(stats.nct.sf(t_a, df, shift) + stats.nct.cdf(-t_a, df, shift))


def sample_size_means(
    mde: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    variance_reduction: float = 0.0,
    t_distribution: bool = False,
    sd_treatment: float | None = None,
) -> SampleSize:
    """Users per arm to detect an absolute difference ``mde`` in means."""
    if mde == 0 or sd <= 0 or (sd_treatment is not None and sd_treatment <= 0):
        raise ValueError("mde must be non-zero and standard deviations positive")
    if not 0.0 <= variance_reduction < 1.0:
        raise ValueError("variance_reduction must lie in [0, 1)")
    sd_t = sd if sd_treatment is None else sd_treatment
    var = (sd**2 + sd_t**2 / ratio) * (1.0 - variance_reduction)
    z_a, z_b = z_quantile(alpha), float(stats.norm.ppf(power))
    n_c = (z_a + z_b) ** 2 * var / mde**2
    if t_distribution:
        n = max(2, math.ceil(n_c))
        while power_means(mde, sd, n, alpha, ratio, variance_reduction, True, sd_t) < power:
            n += 1
        n_c = n
    return _split(n_c, ratio)


def mde_means(
    sd: float,
    n_control: int,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    variance_reduction: float = 0.0,
) -> float:
    """Smallest absolute difference in means detectable with the given power (normal approx.)."""
    z_a, z_b = z_quantile(alpha), float(stats.norm.ppf(power))
    se = (
        sd
        * math.sqrt(1.0 - variance_reduction)
        * math.sqrt(1 / n_control + 1 / (n_control * ratio))
    )
    return (z_a + z_b) * se
