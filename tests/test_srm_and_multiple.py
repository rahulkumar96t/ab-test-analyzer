import numpy as np
import pytest
from scipy import stats

from abtest.multiple import adjust_pvalues
from abtest.srm import srm_check

# Benjamini and Hochberg (1995), section 4: 15 p-values from a clinical trial.
BH_1995 = [
    0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
]  # fmt: skip


def test_srm_matches_scipy_chisquare() -> None:
    res = srm_check([50_500, 49_500])
    ref = stats.chisquare([50_500, 49_500], [50_000, 50_000])
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)


def test_srm_threshold_semantics() -> None:
    # 1 % imbalance on 100k users: chi-square = 2 * 500^2 / 50,000 = 10, p = 0.00157
    res = srm_check([50_500, 49_500])
    assert res.statistic == pytest.approx(10.0)
    assert res.p_value == pytest.approx(0.001565, rel=1e-3)
    assert not res.mismatch
    assert srm_check([50_500, 49_500], threshold=0.01).mismatch


def test_srm_balanced_and_custom_split() -> None:
    assert not srm_check([10_020, 9_980]).mismatch
    res = srm_check([20_000, 80_000], expected_shares=[0.2, 0.8])
    assert res.statistic == pytest.approx(0.0)
    assert not res.mismatch
    assert srm_check([20_000, 80_000]).mismatch
    assert res.observed_shares == pytest.approx((0.2, 0.8))


def test_srm_three_arms_and_errors() -> None:
    res = srm_check([1000, 1000, 1300])
    assert res.mismatch
    with pytest.raises(ValueError):
        srm_check([10])
    with pytest.raises(ValueError):
        srm_check([10, 10], expected_shares=[1.0])


def test_bh_1995_example_rejections() -> None:
    p = np.array(BH_1995)
    assert (adjust_pvalues(p, "bh") < 0.05).sum() == 4
    assert (adjust_pvalues(p, "bonferroni") < 0.05).sum() == 3
    assert (adjust_pvalues(p, "holm") < 0.05).sum() == 3


def test_bh_matches_scipy() -> None:
    p = np.random.default_rng(0).uniform(0, 0.2, 40)
    assert adjust_pvalues(p, "bh") == pytest.approx(stats.false_discovery_control(p, method="bh"))


def test_holm_hand_computed() -> None:
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = .02, .03, .06, .04 -> monotone .06
    assert adjust_pvalues([0.01, 0.04, 0.03, 0.005], "holm") == pytest.approx(
        [0.03, 0.06, 0.06, 0.02]
    )


def test_adjustments_are_never_below_raw() -> None:
    p = np.random.default_rng(1).uniform(size=25)
    for method in ("bonferroni", "holm", "bh"):
        adj = adjust_pvalues(p, method)
        assert (adj >= p - 1e-15).all()
        assert (adj <= 1).all()
    assert adjust_pvalues(p, "none") == pytest.approx(p)
    with pytest.raises(ValueError):
        adjust_pvalues([0.5, 1.2])
