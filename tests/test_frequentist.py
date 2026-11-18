import math

import numpy as np
import pytest
from scipy import stats

from abtest.frequentist import (
    mann_whitney,
    mann_whitney_pvalues,
    normal_test,
    two_proportion_ztest,
    welch_pvalues,
    welch_ttest,
    welch_ttest_from_stats,
    ztest_pvalues,
)


@pytest.mark.parametrize(("x_c", "n_c", "x_t", "n_t"), [(120, 1000, 150, 1000), (31, 412, 18, 388)])
def test_ztest_squared_equals_pearson_chi_square(x_c: int, n_c: int, x_t: int, n_t: int) -> None:
    res = two_proportion_ztest(x_c, n_c, x_t, n_t)
    chi2, p, _, _ = stats.chi2_contingency([[x_c, n_c - x_c], [x_t, n_t - x_t]], correction=False)
    assert res.statistic**2 == pytest.approx(chi2, rel=1e-10)
    assert res.p_value == pytest.approx(p, rel=1e-10)
    assert res.effect == pytest.approx(x_t / n_t - x_c / n_c)


def test_ztest_hand_computed() -> None:
    # 12 % vs 15 % on 1,000 each: pooled 13.5 %, se = sqrt(.135 * .865 * 2 / 1000)
    res = two_proportion_ztest(120, 1000, 150, 1000)
    assert res.statistic == pytest.approx(0.03 / math.sqrt(0.135 * 0.865 * 0.002), rel=1e-12)
    lo, hi = res.ci
    se = math.sqrt(0.12 * 0.88 / 1000 + 0.15 * 0.85 / 1000)
    assert (lo, hi) == pytest.approx((0.03 - 1.959964 * se, 0.03 + 1.959964 * se), rel=1e-5)
    assert res.relative_effect == pytest.approx(0.25)


def test_ztest_vectorised_matches_scalar() -> None:
    x_c, x_t = np.array([100, 130, 90]), np.array([110, 100, 140])
    vec = ztest_pvalues(x_c, 1000, x_t, 1000)
    scalar = [
        two_proportion_ztest(int(c), 1000, int(t), 1000).p_value
        for c, t in zip(x_c, x_t, strict=True)
    ]
    assert vec == pytest.approx(scalar)


def test_ztest_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        two_proportion_ztest(5, 0, 1, 10)
    with pytest.raises(ValueError):
        two_proportion_ztest(11, 10, 1, 10)


def test_welch_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    control, treatment = rng.normal(10, 2, 80), rng.normal(10.8, 5, 140)
    ours = welch_ttest(control, treatment)
    ref = stats.ttest_ind(treatment, control, equal_var=False)
    assert ours.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    assert ours.df == pytest.approx(ref.df, rel=1e-10)
    ci = ref.confidence_interval(0.95)
    assert ours.ci == pytest.approx((ci.low, ci.high), rel=1e-10)


def test_welch_from_summary_statistics_equals_raw() -> None:
    rng = np.random.default_rng(2)
    c, t = rng.exponential(3, 500), rng.exponential(3.3, 450)
    raw = welch_ttest(c, t)
    summary = welch_ttest_from_stats(
        c.mean(), c.var(ddof=1), c.size, t.mean(), t.var(ddof=1), t.size
    )
    assert summary == raw


def test_welch_vectorised_matches_scipy_along_rows() -> None:
    rng = np.random.default_rng(3)
    c, t = rng.normal(0, 1, (25, 60)), rng.normal(0.2, 1.5, (25, 70))
    ref = stats.ttest_ind(t, c, equal_var=False, axis=-1).pvalue
    assert welch_pvalues(c, t) == pytest.approx(ref, rel=1e-9)


def _zero_inflated(rng: np.random.Generator, size: int) -> np.ndarray:
    values = np.round(rng.lognormal(3, 1, size), 0)
    return np.where(rng.random(size) < 0.8, 0.0, values)  # heavy ties at zero


def test_mann_whitney_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(4)
    c, t = _zero_inflated(rng, 400), _zero_inflated(rng, 350)
    ours = mann_whitney(c, t)
    ref = stats.mannwhitneyu(t, c, method="asymptotic", use_continuity=True)
    assert ours.u_statistic == pytest.approx(ref.statistic)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert 0 < ours.prob_superiority < 1


def test_mann_whitney_vectorised_matches_scipy() -> None:
    rng = np.random.default_rng(5)
    c = np.stack([_zero_inflated(rng, 120) for _ in range(10)])
    t = np.stack([_zero_inflated(rng, 100) for _ in range(10)])
    ref = stats.mannwhitneyu(t, c, method="asymptotic", axis=-1).pvalue
    assert mann_whitney_pvalues(c, t) == pytest.approx(ref, rel=1e-9)


def test_relative_interval_and_normal_test() -> None:
    res = two_proportion_ztest(100, 2000, 130, 2000)
    lo, hi = res.relative_ci
    assert lo < res.relative_effect < hi
    assert normal_test(0.0, 0.0) == (0.0, 1.0)
    z, p = normal_test(1.959964, 1.0)
    assert p == pytest.approx(0.05, abs=1e-6)
    assert z == pytest.approx(1.959964)
