"""Seeded simulator for user-level experiment data with planted effects.

Every user gets a latent engagement level that drives sessions, conversion and the
pre-experiment covariates, so CUPED has something real to exploit and sessions of the same
user are correlated (which is what makes ratio metrics need the delta method). Effects are
relative and applied to the treatment arm only:

* ``conversion_lift``   - multiplies the conversion probability;
* ``order_value_lift``  - multiplies the order value of converters;
* ``sessions_lift``     - multiplies the expected number of extra sessions;
* ``page_load_lift``    - multiplies page load time (a guardrail, positive is worse);
* ``bounce_lift``       - multiplies the bounce probability (a guardrail, positive is worse).

``assigned_treatment_share`` different from ``treatment_share`` plants a sample-ratio mismatch.
The data is synthetic; nothing in it describes real users.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS = (
    "user_id",
    "variant",
    "day",
    "pre_revenue",
    "pre_sessions",
    "sessions",
    "converted",
    "revenue",
    "page_load_ms",
    "bounced",
)


@dataclass(frozen=True)
class ExperimentConfig:
    n_users: int = 40_000
    treatment_share: float = 0.5
    assigned_treatment_share: float | None = None
    days: int = 14
    baseline_conversion: float = 0.08
    conversion_lift: float = 0.0
    mean_order_value: float = 60.0
    order_value_sigma: float = 0.9
    order_value_lift: float = 0.0
    engagement_sd: float = 1.0
    spend_sd: float = 0.6
    sessions_mean: float = 4.0
    sessions_lift: float = 0.0
    pre_period_days: int = 56
    page_load_ms: float = 900.0
    page_load_lift: float = 0.0
    bounce_rate: float = 0.30
    bounce_lift: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_users < 2:
            raise ValueError("n_users must be at least 2")
        share = self.actual_share
        if not 0.0 < share < 1.0:
            raise ValueError("treatment shares must lie strictly between 0 and 1")
        if not 0.0 < self.baseline_conversion < 1.0 or not 0.0 < self.bounce_rate < 1.0:
            raise ValueError("rates must lie strictly between 0 and 1")
        if self.sessions_mean < 1.0 or self.days < 1 or self.pre_period_days < 1:
            raise ValueError("sessions_mean must be >= 1 and periods at least one day")
        if not 0.0 <= self.spend_sd < self.order_value_sigma:
            raise ValueError("spend_sd must lie in [0, order_value_sigma)")

    @property
    def actual_share(self) -> float:
        if self.assigned_treatment_share is None:
            return self.treatment_share
        return self.assigned_treatment_share


def _mean_one_lognormal(rng: np.random.Generator, sigma: float, size: int) -> np.ndarray:
    return rng.lognormal(-(sigma**2) / 2.0, sigma, size)


def simulate_experiment(config: ExperimentConfig | None = None) -> pd.DataFrame:
    """One row per user; the same config and seed always give the same frame."""
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users

    is_t = rng.random(n) < cfg.actual_share
    day = rng.integers(1, cfg.days + 1, n)
    # Latent user traits, both with mean 1: how active a user is and how much they spend.
    engagement = _mean_one_lognormal(rng, cfg.engagement_sd, n)
    spend = _mean_one_lognormal(rng, cfg.spend_sd, n)
    order_sigma = float(np.sqrt(cfg.order_value_sigma**2 - cfg.spend_sd**2))

    # Pre-experiment period (no treatment): the CUPED covariates.
    scale = cfg.pre_period_days / cfg.days
    pre_sessions = rng.poisson(scale * (cfg.sessions_mean - 1.0) * engagement)
    pre_orders = rng.poisson(scale * cfg.baseline_conversion * engagement)
    owner = np.repeat(np.arange(n), pre_orders)
    pre_values = (
        cfg.mean_order_value * spend[owner] * _mean_one_lognormal(rng, order_sigma, owner.size)
    )
    pre_revenue = np.bincount(owner, weights=pre_values, minlength=n)

    # Experiment period.
    sessions_rate = (
        (cfg.sessions_mean - 1.0) * engagement * np.where(is_t, 1 + cfg.sessions_lift, 1)
    )
    sessions = 1 + rng.poisson(sessions_rate)
    p_convert = cfg.baseline_conversion * engagement * np.where(is_t, 1 + cfg.conversion_lift, 1)
    converted = rng.random(n) < np.clip(p_convert, 0.0, 0.95)
    order_value = cfg.mean_order_value * spend * _mean_one_lognormal(rng, order_sigma, n)
    order_value *= np.where(is_t, 1 + cfg.order_value_lift, 1)
    revenue = np.where(converted, order_value, 0.0)

    page_load = cfg.page_load_ms * _mean_one_lognormal(rng, 0.35, n)
    page_load *= np.where(is_t, 1 + cfg.page_load_lift, 1)
    p_bounce = np.clip(cfg.bounce_rate * np.where(is_t, 1 + cfg.bounce_lift, 1), 0.0, 1.0)
    bounced = rng.random(n) < p_bounce

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": np.where(is_t, "treatment", "control"),
            "day": day,
            "pre_revenue": np.round(pre_revenue, 2),
            "pre_sessions": pre_sessions,
            "sessions": sessions,
            "converted": converted.astype(int),
            "revenue": np.round(revenue, 2),
            "page_load_ms": np.round(page_load, 1),
            "bounced": bounced.astype(int),
        },
        columns=list(COLUMNS),
    )
