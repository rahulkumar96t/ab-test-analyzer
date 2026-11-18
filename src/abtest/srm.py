"""Sample-ratio mismatch (SRM) check.

If users were assigned 50/50 but 50,800 / 49,200 arrived, something between assignment and
logging is broken (a redirect that loses users, a bot filter applied to one arm, a crash).
Effects measured on such data are not trustworthy, whatever their p-values say. The check is a
chi-square goodness-of-fit test of the observed counts against the configured split. A strict
threshold (0.001 by default) is common because the check runs on every experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    statistic: float
    p_value: float
    threshold: float

    @property
    def mismatch(self) -> bool:
        return self.p_value < self.threshold

    @property
    def observed_shares(self) -> tuple[float, ...]:
        total = sum(self.observed)
        return tuple(o / total for o in self.observed)


def srm_check(
    counts: Sequence[int], expected_shares: Sequence[float] | None = None, threshold: float = 0.001
) -> SRMResult:
    """Chi-square goodness-of-fit test of arm sizes against the intended allocation."""
    observed = np.asarray(counts, dtype=float)
    if observed.ndim != 1 or observed.size < 2:
        raise ValueError("need the counts of at least two arms")
    if (observed < 0).any() or observed.sum() == 0:
        raise ValueError("counts must be non-negative and not all zero")
    shares = (
        np.full(observed.size, 1.0 / observed.size)
        if expected_shares is None
        else np.asarray(expected_shares, dtype=float)
    )
    if shares.shape != observed.shape or (shares <= 0).any():
        raise ValueError("expected_shares must be positive, one per arm")
    shares = shares / shares.sum()
    expected = observed.sum() * shares
    statistic = float(((observed - expected) ** 2 / expected).sum())
    p_value = float(stats.chi2.sf(statistic, df=observed.size - 1))
    return SRMResult(
        observed=tuple(int(c) for c in counts),
        expected=tuple(float(e) for e in expected),
        statistic=statistic,
        p_value=p_value,
        threshold=threshold,
    )
