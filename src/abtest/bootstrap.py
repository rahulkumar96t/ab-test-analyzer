"""Two-sample bootstrap confidence intervals for any statistic.

Each arm is resampled independently with its own size, and a unit (user) is resampled together
with all its columns, so ratio metrics keep the numerator / denominator pairing. Resamples are
generated in batches to bound memory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Statistic = Callable[..., NDArray[np.float64]]
BootstrapMethod = Literal["percentile", "basic"]


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_resamples: int
    method: BootstrapMethod


def mean_statistic(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values.mean(axis=-1)


def ratio_statistic(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    return numerator.sum(axis=-1) / denominator.sum(axis=-1)


def _as_columns(arm: Sequence[ArrayLike]) -> list[NDArray[np.float64]]:
    cols = [np.asarray(c, dtype=float) for c in arm]
    if not cols or any(c.ndim != 1 or c.shape != cols[0].shape for c in cols):
        raise ValueError("each arm needs one or more 1-D columns of equal length")
    return cols


def _resampled_statistic(
    cols: list[NDArray[np.float64]],
    statistic: Statistic,
    n_resamples: int,
    rng: np.random.Generator,
    max_cells: int,
) -> NDArray[np.float64]:
    n = cols[0].size
    batch = max(1, max_cells // max(n, 1))
    out = np.empty(n_resamples)
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        out[start : start + size] = statistic(*(c[idx] for c in cols))
    return out


def bootstrap_difference(
    control: Sequence[ArrayLike],
    treatment: Sequence[ArrayLike],
    statistic: Statistic = mean_statistic,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    method: BootstrapMethod = "percentile",
    seed: int | None = 0,
    max_cells: int = 4_000_000,
) -> BootstrapResult:
    """CI for ``statistic(treatment) - statistic(control)``.

    ``control`` and ``treatment`` are sequences of columns, e.g. ``(revenue,)`` or
    ``(revenue, sessions)``; ``statistic`` receives the resampled columns as 2-D arrays
    (resample x unit) and must reduce over the last axis.
    """
    if n_resamples < 100:
        raise ValueError("use at least 100 resamples")
    c_cols, t_cols = _as_columns(control), _as_columns(treatment)
    rng = np.random.default_rng(seed)
    observed = float(
        statistic(*(c[None, :] for c in t_cols))[0] - statistic(*(c[None, :] for c in c_cols))[0]
    )
    diffs = _resampled_statistic(t_cols, statistic, n_resamples, rng, max_cells)
    diffs -= _resampled_statistic(c_cols, statistic, n_resamples, rng, max_cells)
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    if method == "basic":
        lo, hi = 2 * observed - hi, 2 * observed - lo
    elif method != "percentile":
        raise ValueError(f"unknown bootstrap method {method!r}")
    return BootstrapResult(
        estimate=observed,
        ci_low=float(lo),
        ci_high=float(hi),
        se=float(diffs.std(ddof=1)),
        n_resamples=n_resamples,
        method=method,
    )
