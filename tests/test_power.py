import numpy as np
import pytest

from abtest.frequentist import ztest_pvalues
from abtest.power import (
    mde_means,
    mde_proportions,
    power_means,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)


@pytest.mark.parametrize(("d", "n"), [(0.5, 64), (0.8, 26), (0.2, 394)])
def test_two_sample_t_sizes_match_published_values(d: float, n: int) -> None:
    # Cohen (1988) power tables: 64 and 26 per group for d = 0.5 / 0.8 at alpha .05, power .80.
    # For d = 0.2 the exact noncentral-t answer is 394 (Cohen's approximate table lists 393).
    assert sample_size_means(d, 1.0, t_distribution=True).control == n


def test_normal_approximation_for_d_half() -> None:
    # 2 * (1.95996 + 0.84162)^2 / 0.25 = 62.8 -> 63 per group
    assert sample_size_means(0.5, 1.0).control == 63
    assert power_means(0.5, 1.0, 63) >= 0.8 > power_means(0.5, 1.0, 62)


def test_proportion_sample_size_is_the_smallest_with_enough_power() -> None:
    n = sample_size_proportions(0.10, 0.02).control
    assert power_proportions(0.10, 0.02, n) >= 0.8
    assert power_proportions(0.10, 0.02, n - 1) < 0.8


def test_relative_and_absolute_effects_agree() -> None:
    assert sample_size_proportions(0.05, 0.10, relative=True) == sample_size_proportions(
        0.05, 0.005
    )


def test_mde_roundtrips() -> None:
    n = sample_size_proportions(0.08, 0.01).control
    assert mde_proportions(0.08, n) == pytest.approx(0.01, rel=1e-3)
    assert mde_proportions(0.08, n, relative=True) == pytest.approx(0.125, rel=1e-3)
    n_means = sample_size_means(2.0, 15.0).control
    assert mde_means(15.0, n_means) == pytest.approx(2.0, rel=5e-3)


def test_variance_reduction_and_allocation() -> None:
    full = sample_size_means(1.0, 10.0).control
    half = sample_size_means(1.0, 10.0, variance_reduction=0.5).control
    assert half == pytest.approx(full / 2, abs=1)
    even = sample_size_proportions(0.1, 0.02)
    skewed = sample_size_proportions(0.1, 0.02, ratio=3.0)
    assert skewed.treatment == pytest.approx(3 * skewed.control, abs=1)
    assert skewed.total > even.total  # unequal splits cost sample


def test_treatment_sd_lowers_power_and_raises_sample_size() -> None:
    assert power_means(1.0, 10.0, 500, sd_treatment=10.0) == power_means(1.0, 10.0, 500)
    assert power_means(1.0, 10.0, 500, sd_treatment=12.0) < power_means(1.0, 10.0, 500)
    equal = sample_size_means(1.0, 10.0).control
    wider = sample_size_means(1.0, 10.0, sd_treatment=12.0).control
    # n is proportional to sd_c^2 + sd_t^2: (100 + 144) / 200
    assert wider == pytest.approx(equal * 244 / 200, abs=1)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        sample_size_proportions(0.1, 0.0)
    with pytest.raises(ValueError):
        sample_size_proportions(0.95, 0.10)
    with pytest.raises(ValueError):
        sample_size_means(1.0, 1.0, variance_reduction=1.0)
    with pytest.raises(ValueError):
        mde_proportions(0.5, 3, power=0.99)


def test_planned_power_is_reached_in_simulation() -> None:
    rng = np.random.default_rng(11)
    n = sample_size_proportions(0.10, 0.10, relative=True).control
    sims = 6000
    p = ztest_pvalues(rng.binomial(n, 0.10, sims), n, rng.binomial(n, 0.11, sims), n)
    assert abs((p < 0.05).mean() - 0.80) < 0.025  # MC standard error is about 0.005
