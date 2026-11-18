"""Small, seeded Monte-Carlo checks of the false-positive rate (fast enough for CI)."""

import numpy as np
import pytest

from abtest.frequentist import mann_whitney_pvalues, welch_pvalues, ztest_pvalues
from abtest.validation import Rate, run_validation, simulate_peeking


def test_ztest_false_positive_rate_is_nominal() -> None:
    rng = np.random.default_rng(20261117)
    sims, n = 40_000, 5_000
    p = ztest_pvalues(rng.binomial(n, 0.1, sims), n, rng.binomial(n, 0.1, sims), n)
    assert abs((p < 0.05).mean() - 0.05) < 0.005  # MC standard error ~0.0011


def test_welch_and_mann_whitney_false_positive_rate_on_skewed_data() -> None:
    rng = np.random.default_rng(7)
    shape = (3000, 400)
    c = np.where(rng.random(shape) < 0.3, rng.lognormal(3, 1.2, shape), 0.0)
    t = np.where(rng.random(shape) < 0.3, rng.lognormal(3, 1.2, shape), 0.0)
    assert abs((welch_pvalues(c, t) < 0.05).mean() - 0.05) < 0.015
    assert abs((mann_whitney_pvalues(c, t) < 0.05).mean() - 0.05) < 0.015


def test_peeking_inflates_errors_and_sequential_methods_do_not() -> None:
    row = simulate_peeking(sims=6000, looks=20, seed=3)
    assert row.naive.value > 0.18
    assert row.fixed.value == pytest.approx(0.05, abs=0.012)
    assert row.gst.value == pytest.approx(0.05, abs=0.012)
    assert row.msprt.value <= 0.05 + 0.006
    with pytest.raises(ValueError):
        simulate_peeking(sims=10, looks=7, n_per_arm=100)


def test_sequential_methods_keep_power_and_stop_early() -> None:
    row = simulate_peeking(sims=3000, looks=10, lift=0.10, seed=4)
    assert row.gst.value > 0.8
    assert row.gst_mean_fraction < 0.8
    assert row.msprt.value > 0.5


def test_wilson_interval() -> None:
    r = Rate(50, 1000)
    lo, hi = r.wilson()
    assert lo < 0.05 < hi
    assert (lo, hi) == pytest.approx((0.0381, 0.0653), abs=5e-4)
    assert Rate(0, 100).wilson()[0] == 0.0


def test_run_validation_subset_is_consistent() -> None:
    report = run_validation(sims=200, seed=1, only=["ztest", "ratio"])
    by_name = {c.procedure: c for c in report.checks}
    assert by_name["delta method (user-level)"].measured.value < 0.10
    assert by_name["naive page-view-level z-test"].measured.value > 0.15
    assert report.extras["ztest_n_per_arm"] > 50_000
