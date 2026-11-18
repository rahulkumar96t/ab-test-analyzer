import math

import numpy as np
import pytest
from scipy import special, stats

from abtest.bayes import (
    beta_binomial,
    expected_shortfall_beta,
    normal_posterior,
    prob_beta_greater,
)


def closed_form_prob_greater(a_t: int, b_t: int, a_c: int, b_c: int) -> float:
    """P(T > C) for integer a_t (Evan Miller, "Formulas for Bayesian A/B Testing")."""
    i = np.arange(a_t)
    log_terms = (
        special.betaln(a_c + i, b_c + b_t)
        - np.log(b_t + i)
        - special.betaln(1 + i, b_t)
        - special.betaln(a_c, b_c)
    )
    return float(np.exp(log_terms).sum())


@pytest.mark.parametrize(
    "params", [(3, 9, 2, 10), (21, 181, 15, 187), (1 + 480, 1 + 4520, 1 + 430, 1 + 4570)]
)
def test_probability_matches_closed_form(params: tuple[int, int, int, int]) -> None:
    assert prob_beta_greater(*params) == pytest.approx(closed_form_prob_greater(*params), abs=1e-7)


def test_probability_and_loss_match_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    a_t, b_t, a_c, b_c = 1 + 130, 1 + 870, 1 + 110, 1 + 890
    t, c = rng.beta(a_t, b_t, 2_000_000), rng.beta(a_c, b_c, 2_000_000)
    assert prob_beta_greater(a_t, b_t, a_c, b_c) == pytest.approx((t > c).mean(), abs=1.5e-3)
    loss = expected_shortfall_beta(a_t, b_t, a_c, b_c)
    assert loss == pytest.approx(np.maximum(c - t, 0).mean(), rel=0.02)


def test_symmetric_data_gives_even_odds() -> None:
    res = beta_binomial(100, 1000, 100, 1000)
    assert res.prob_better == pytest.approx(0.5, abs=1e-9)
    assert res.expected_loss_ship == pytest.approx(res.expected_loss_keep, abs=1e-12)
    assert res.ci_low < 0 < res.ci_high


def test_loss_identity_and_direction_flip() -> None:
    res = beta_binomial(90, 1000, 120, 1000)
    assert res.expected_loss_keep - res.expected_loss_ship == pytest.approx(res.diff_mean)
    assert res.prob_better > 0.95
    flipped = res.lower_is_better()
    assert flipped.prob_better == pytest.approx(1 - res.prob_better)
    assert flipped.expected_loss_ship == res.expected_loss_keep


def test_normal_posterior_flat_prior() -> None:
    res = normal_posterior(effect=0.3, se=0.2)
    assert res.prob_better == pytest.approx(stats.norm.cdf(1.5))
    assert (res.ci_low, res.ci_high) == pytest.approx((0.3 - 1.959964 * 0.2, 0.3 + 1.959964 * 0.2))
    draws = np.random.default_rng(1).normal(0.3, 0.2, 2_000_000)
    assert res.expected_loss_ship == pytest.approx(np.maximum(-draws, 0).mean(), rel=0.01)
    assert res.expected_loss_keep == pytest.approx(np.maximum(draws, 0).mean(), rel=0.01)


def test_normal_prior_shrinks_towards_zero() -> None:
    flat = normal_posterior(0.3, 0.2)
    shrunk = normal_posterior(0.3, 0.2, prior_sd=0.2)
    assert shrunk.diff_mean == pytest.approx(0.15)  # equal precisions -> halfway
    assert shrunk.ci_high - shrunk.ci_low == pytest.approx(
        (flat.ci_high - flat.ci_low) / math.sqrt(2)
    )
    with pytest.raises(ValueError):
        normal_posterior(0.1, 0.0)
