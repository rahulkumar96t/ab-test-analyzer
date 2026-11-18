import numpy as np
import pytest
from scipy import stats

from abtest.sequential import (
    alpha_spent,
    group_sequential_design,
    msprt_half_width,
    msprt_log_lambda,
    msprt_monitor,
)

FIVE_LOOKS = [0.2, 0.4, 0.6, 0.8, 1.0]


def test_single_look_is_the_fixed_horizon_test() -> None:
    assert group_sequential_design([1.0]).boundaries[0] == pytest.approx(1.959964, abs=1e-6)


def test_reproduces_published_obrien_fleming_boundaries() -> None:
    # gsDesign's symmetric design with sfLDOF and one-sided alpha 0.025 per side, five equally
    # spaced looks, has efficacy bounds 4.877, 3.357, 2.680, 2.290, 2.031.
    def per_side_ldof(t: np.ndarray) -> np.ndarray:
        return 2 * (2 - 2 * stats.norm.cdf(stats.norm.isf(0.0125) / np.sqrt(t)))

    design = group_sequential_design(FIVE_LOOKS, spending=per_side_ldof)
    assert design.boundaries == pytest.approx((4.877, 3.357, 2.680, 2.290, 2.031), abs=1e-3)
    assert design.spending == "custom"


def test_pocock_type_boundaries_are_nearly_flat() -> None:
    b = group_sequential_design(FIVE_LOOKS, spending="pocock").boundaries
    assert max(b) - min(b) < 0.06
    assert b[0] == pytest.approx(stats.norm.isf(0.05 * np.log1p((np.e - 1) * 0.2) / 2))


def test_boundaries_spend_exactly_alpha_in_simulation() -> None:
    design = group_sequential_design(FIVE_LOOKS)
    rng = np.random.default_rng(0)
    paths = (rng.standard_normal((400_000, 5)) * np.sqrt(0.2)).cumsum(axis=1)
    z = paths / np.sqrt(FIVE_LOOKS)
    crossed = (np.abs(z) >= np.array(design.boundaries)).any(axis=1).mean()
    assert crossed == pytest.approx(0.05, abs=0.0015)
    assert design.first_crossing([0.5, 3.2, 0.1]) == 2
    assert design.first_crossing([0.5, 0.5]) is None


def test_alpha_spending_functions() -> None:
    for kind in ("obf", "pocock"):
        spent = alpha_spent(np.linspace(0.01, 1, 50), 0.05, kind)
        assert (np.diff(spent) >= 0).all()  # OBF spends ~0 (underflows) at tiny fractions
        assert (np.diff(spent[10:]) > 0).all()
        assert spent[-1] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        group_sequential_design([0.5, 0.4])


def test_msprt_controls_type_one_error_under_continuous_monitoring() -> None:
    rng = np.random.default_rng(1)
    sims, looks, batch = 3000, 200, 50
    # difference of two arms' means of N(0, 1) data, observed after every batch
    sums = (rng.standard_normal((sims, looks)) * np.sqrt(2 * batch)).cumsum(axis=1)
    n = batch * np.arange(1, looks + 1)
    diff, var = sums / n, 2.0 / n
    msprt = (msprt_log_lambda(diff, var, tau=0.1) >= np.log(20)).any(axis=1).mean()
    naive = (np.abs(diff / np.sqrt(var)) >= 1.959964).any(axis=1).mean()
    assert msprt <= 0.05 + 0.008  # Ville's inequality: at most alpha
    assert naive > 0.25  # peeking 200 times at a fixed-horizon test


def test_msprt_detects_a_real_effect_and_its_interval_covers_it() -> None:
    rng = np.random.default_rng(2)
    n = np.arange(1, 41) * 100
    # each batch of 100 per arm adds N(0.2 * 100, 2 * 100) to the summed difference
    diff = (0.2 * 100 + rng.standard_normal(40) * np.sqrt(200)).cumsum() / n
    summary = msprt_monitor(diff.tolist(), (2 / n).tolist(), tau=0.2)
    assert summary.p_value < 0.05
    assert summary.first_rejection is not None
    assert summary.ci_low <= 0.2 <= summary.ci_high
    assert all(a >= b for a, b in zip(summary.p_values, summary.p_values[1:], strict=False))


def test_always_valid_interval_is_wider_than_fixed_and_shrinks() -> None:
    v = np.array([0.04, 0.01, 0.0025])
    half = msprt_half_width(v, tau=0.1, alpha=0.05)
    assert (half > 1.96 * np.sqrt(v)).all()
    assert (np.diff(half) < 0).all()
    with pytest.raises(ValueError):
        msprt_monitor([0.1], [0.01], tau=0.0)
