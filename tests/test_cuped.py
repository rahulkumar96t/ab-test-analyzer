import numpy as np
import pytest

from abtest.cuped import cuped_adjust, cuped_test, cuped_theta


def _arms(
    rng: np.random.Generator, n: int, effect: float, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-period x and metric y with corr(x, y) = rho in each arm, effect added to treatment."""
    x = rng.normal(0.0, 1.0, (2, n))
    y = 5.0 + rho * x + np.sqrt(1 - rho**2) * rng.normal(0.0, 1.0, (2, n))
    y[1] += effect
    return y[0], x[0], y[1], x[1]


def test_theta_is_the_ols_slope() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.0 + 0.7 * x + rng.normal(size=500)
    assert cuped_theta(y, x) == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-10)
    assert cuped_theta(y, np.ones(500)) == 0.0


def test_adjustment_preserves_the_mean_and_ignores_constant_covariate() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=100), rng.normal(size=100)
    assert cuped_adjust(y, x, 0.8).mean() == pytest.approx(y.mean())
    res = cuped_test(y[:50], np.ones(50), y[50:], np.ones(50))
    assert res.adjusted.effect == pytest.approx(res.raw.effect)
    assert res.variance_reduction == pytest.approx(0.0, abs=1e-12)


def test_variance_reduction_is_about_rho_squared() -> None:
    rng = np.random.default_rng(2)
    res = cuped_test(*_arms(rng, 20_000, 0.0, rho=0.8))
    assert res.correlation == pytest.approx(0.8, abs=0.01)
    assert res.variance_reduction == pytest.approx(0.64, abs=0.02)


def test_cuped_is_unbiased_and_tighter_in_monte_carlo() -> None:
    rng = np.random.default_rng(3)
    effect, sims = 0.10, 600
    raw, adj = np.empty(sims), np.empty(sims)
    for i in range(sims):
        res = cuped_test(*_arms(rng, 400, effect, rho=0.7))
        raw[i], adj[i] = res.raw.effect, res.adjusted.effect
    mc_se = adj.std(ddof=1) / np.sqrt(sims)
    assert abs(adj.mean() - effect) < 4 * mc_se  # unbiased
    assert abs(raw.mean() - effect) < 4 * raw.std(ddof=1) / np.sqrt(sims)
    reduction = 1 - adj.var(ddof=1) / raw.var(ddof=1)
    assert reduction == pytest.approx(0.49, abs=0.08)  # rho^2 = 0.49


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cuped_test(np.ones(5), np.ones(4), np.ones(5), np.ones(5))
