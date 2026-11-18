import numpy as np
import pytest
from scipy import stats

from abtest.bootstrap import bootstrap_difference, mean_statistic, ratio_statistic
from abtest.frequentist import welch_ttest
from abtest.ratio import delta_ratio_pvalues, delta_ratio_test, ratio_estimate


def _sessions(rng: np.random.Generator, users: int) -> tuple[np.ndarray, np.ndarray]:
    sessions = 1 + rng.poisson(3 * rng.lognormal(-0.3, 0.8, users))
    revenue = rng.gamma(2.0, 1.5, users) * sessions * rng.uniform(0.5, 1.5, users)
    return revenue, sessions.astype(float)


def test_unit_denominator_reduces_to_mean_and_standard_error() -> None:
    y = np.random.default_rng(0).normal(10, 3, 300)
    ratio, se = ratio_estimate(y, np.ones_like(y))
    assert ratio == pytest.approx(y.mean())
    assert se == pytest.approx(y.std(ddof=1) / np.sqrt(y.size))


def test_delta_standard_error_matches_bootstrap() -> None:
    rng = np.random.default_rng(1)
    rev, ses = _sessions(rng, 3000)
    _, se = ratio_estimate(rev, ses)
    idx = rng.integers(0, rev.size, (2000, rev.size))
    boot = rev[idx].sum(1) / ses[idx].sum(1)
    assert se == pytest.approx(boot.std(ddof=1), rel=0.08)


def test_vectorised_delta_matches_scalar() -> None:
    rng = np.random.default_rng(2)
    rc, sc = _sessions(rng, 500)
    rt, st = _sessions(rng, 500)
    scalar = delta_ratio_test(rc, sc, rt, st).p_value
    assert delta_ratio_pvalues(rc[None], sc[None], rt[None], st[None])[0] == pytest.approx(scalar)
    with pytest.raises(ValueError):
        ratio_estimate(rc, np.zeros_like(sc))


def test_bootstrap_is_reproducible_and_seed_dependent() -> None:
    rng = np.random.default_rng(3)
    c, t = rng.exponential(1, 300), rng.exponential(1.2, 300)
    a = bootstrap_difference([c], [t], n_resamples=500, seed=1)
    b = bootstrap_difference([c], [t], n_resamples=500, seed=1)
    other = bootstrap_difference([c], [t], n_resamples=500, seed=2)
    assert a == b
    assert a.ci_low != other.ci_low
    assert a.estimate == pytest.approx(t.mean() - c.mean())


def test_bootstrap_agrees_with_welch_and_scipy_on_a_large_sample() -> None:
    rng = np.random.default_rng(4)
    c, t = rng.normal(5, 2, 3000), rng.normal(5.2, 2, 3000)
    ours = bootstrap_difference([c], [t], n_resamples=4000, seed=0, max_cells=1_000_000)
    lo, hi = welch_ttest(c, t).ci
    assert ours.ci_low == pytest.approx(lo, abs=0.02)
    assert ours.ci_high == pytest.approx(hi, abs=0.02)
    ref = stats.bootstrap(
        (t, c),
        lambda x, y, axis: x.mean(axis) - y.mean(axis),
        n_resamples=4000,
        method="percentile",
        rng=np.random.default_rng(0),
    )
    assert ours.ci_low == pytest.approx(ref.confidence_interval.low, abs=0.02)
    assert ours.se == pytest.approx(ref.standard_error, rel=0.05)


def test_ratio_bootstrap_and_basic_interval() -> None:
    rng = np.random.default_rng(5)
    rc, sc = _sessions(rng, 800)
    rt, st = _sessions(rng, 800)
    pct = bootstrap_difference([rc, sc], [rt, st], ratio_statistic, n_resamples=1000, seed=3)
    basic = bootstrap_difference(
        [rc, sc], [rt, st], ratio_statistic, n_resamples=1000, seed=3, method="basic"
    )
    delta = delta_ratio_test(rc, sc, rt, st)
    assert pct.estimate == pytest.approx(delta.effect)
    assert pct.se == pytest.approx(delta.se, rel=0.12)
    # the basic interval reflects the percentile interval around the estimate
    assert basic.ci_low == pytest.approx(2 * pct.estimate - pct.ci_high)
    assert basic.ci_high == pytest.approx(2 * pct.estimate - pct.ci_low)


def test_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_difference([np.ones(5)], [np.ones(5)], n_resamples=10)
    with pytest.raises(ValueError):
        bootstrap_difference([np.ones(5), np.ones(4)], [np.ones(5)], mean_statistic)
