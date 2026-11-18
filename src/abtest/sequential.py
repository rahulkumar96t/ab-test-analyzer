"""Sequential testing: mSPRT (always-valid) and group-sequential alpha spending.

Looking at a fixed-horizon test every day and stopping at the first p < 0.05 inflates the false
positive rate far above 5 % (``abtest peeking`` measures how far). Two remedies are implemented:

* **mSPRT** (Johari, Koomen, Pekelis and Walsh, 2017). For an asymptotically normal estimate
  ``d`` with variance ``V`` and a N(0, tau^2) mixture over the alternative, the likelihood
  ratio is

      Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 d^2 / (2 V (V + tau^2)))

  and is a non-negative martingale under H0. By Ville's inequality P(sup Lambda >= 1/alpha) <=
  alpha, so the test may be checked after every observation. ``1 / max Lambda`` is an
  always-valid p-value.

* **Group-sequential design** with a Lan-DeMets spending function: a fixed number of planned
  looks at given information fractions; the boundaries are computed by recursive numerical
  integration so that the probability of crossing under H0 equals the alpha spent at each look.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

Floats = NDArray[np.float64]
Spending = Literal["obf", "pocock"]

# ----------------------------------------------------------------------------------- mSPRT


def msprt_log_lambda(effect: ArrayLike, variance: ArrayLike, tau: float) -> Floats:
    """log of the normal-mixture likelihood ratio, vectorised."""
    d, v = np.asarray(effect, dtype=float), np.asarray(variance, dtype=float)
    t2 = tau**2
    return 0.5 * np.log(v / (v + t2)) + t2 * d**2 / (2.0 * v * (v + t2))


def msprt_half_width(variance: ArrayLike, tau: float, alpha: float) -> Floats:
    """Half-width of the always-valid confidence interval (the set where Lambda < 1/alpha)."""
    v = np.asarray(variance, dtype=float)
    t2 = tau**2
    return np.sqrt(v * (v + t2) / t2 * (2.0 * math.log(1.0 / alpha) + np.log((v + t2) / v)))


@dataclass(frozen=True)
class SequentialSummary:
    """mSPRT applied to a sequence of looks at accumulating data."""

    looks: int
    p_values: tuple[float, ...]
    ci_low: float
    ci_high: float
    tau: float
    alpha: float

    @property
    def p_value(self) -> float:
        return self.p_values[-1]

    @property
    def first_rejection(self) -> int | None:
        """1-based index of the first look with an always-valid p-value below alpha."""
        for i, p in enumerate(self.p_values, start=1):
            if p < self.alpha:
                return i
        return None


def msprt_monitor(
    effects: Sequence[float], variances: Sequence[float], tau: float, alpha: float = 0.05
) -> SequentialSummary:
    """Always-valid p-values and the running-intersection confidence sequence over the looks."""
    if len(effects) != len(variances) or not effects:
        raise ValueError("need one variance per effect and at least one look")
    if tau <= 0:
        raise ValueError("tau must be positive")
    d, v = np.asarray(effects, dtype=float), np.asarray(variances, dtype=float)
    p = np.minimum.accumulate(np.minimum(1.0, np.exp(-msprt_log_lambda(d, v, tau))))
    half = msprt_half_width(v, tau, alpha)
    return SequentialSummary(
        looks=d.size,
        p_values=tuple(float(x) for x in p),
        ci_low=float(np.max(d - half)),
        ci_high=float(np.min(d + half)),
        tau=tau,
        alpha=alpha,
    )


# ------------------------------------------------------------------ group-sequential design


def alpha_spent(info: ArrayLike, alpha: float, spending: Spending = "obf") -> Floats:
    """Cumulative two-sided alpha spent at information fractions ``info`` (Lan-DeMets)."""
    t = np.clip(np.asarray(info, dtype=float), 0.0, 1.0)
    if spending == "obf":
        with np.errstate(divide="ignore"):
            out = 2.0 - 2.0 * stats.norm.cdf(stats.norm.isf(alpha / 2.0) / np.sqrt(t))
        return np.where(t > 0, out, 0.0)
    if spending == "pocock":
        return alpha * np.log1p((math.e - 1.0) * t)
    raise ValueError(f"unknown spending function {spending!r}")


def _simpson_weights(n: int, h: float) -> Floats:
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return w * h / 3.0


def _crossing_gap(
    c: float, sqrt_t: float, step: float, nodes: Floats, mass: Floats, target: float
) -> float:
    """P(first crossing at this look | boundary c) minus the alpha to spend at this look."""
    b = c * sqrt_t
    tails = stats.norm.cdf((-b - nodes) / step) + stats.norm.sf((b - nodes) / step)
    return float(mass @ tails) - target


@dataclass(frozen=True)
class GroupSequentialDesign:
    info_fractions: tuple[float, ...]
    boundaries: tuple[float, ...]
    alpha: float
    spending: str

    def first_crossing(self, z_values: Sequence[float]) -> int | None:
        """1-based index of the first look whose |z| reaches the boundary."""
        for i, (z, c) in enumerate(zip(z_values, self.boundaries, strict=False), start=1):
            if abs(z) >= c:
                return i
        return None


def group_sequential_design(
    info_fractions: Sequence[float],
    alpha: float = 0.05,
    spending: Spending | Callable[[Floats], Floats] = "obf",
    grid_points: int = 1201,
) -> GroupSequentialDesign:
    """Two-sided symmetric z boundaries for planned looks.

    ``spending`` is ``"obf"`` / ``"pocock"`` (Lan-DeMets, applied to the two-sided ``alpha``)
    or any callable mapping information fractions to cumulative two-sided alpha.

    Works with the score process S_k = Z_k sqrt(t_k), a Brownian motion in information time
    under H0. The sub-density of S on the continuation region is propagated look by look with a
    Gaussian kernel and Simpson's rule; each boundary is the root of "probability of first
    crossing at this look = alpha spent at this look".
    """
    t = np.asarray(info_fractions, dtype=float)
    if t.ndim != 1 or t.size == 0 or (np.diff(t) <= 0).any() or t[0] <= 0 or t[-1] > 1:
        raise ValueError("information fractions must be increasing in (0, 1]")
    if grid_points % 2 == 0:
        grid_points += 1
    spent = spending(t) if callable(spending) else alpha_spent(t, alpha, spending)
    increments = np.diff(np.concatenate([[0.0], np.asarray(spent, dtype=float)]))
    cap = 8.5  # |Z| beyond this has probability < 1e-16 and is treated as unreachable

    boundaries: list[float] = []
    nodes = weights = density = np.empty(0)
    for k, (t_k, spend) in enumerate(zip(t, increments, strict=True)):
        if k == 0:
            c = float(stats.norm.isf(spend / 2.0)) if spend > 1e-15 else math.inf
        else:
            step = math.sqrt(t_k - t[k - 1])
            if spend <= 1e-15:
                c = math.inf
            else:
                args = (math.sqrt(t_k), step, nodes, weights * density, spend)
                c = optimize.brentq(_crossing_gap, 1e-6, 50.0, args=args, xtol=1e-10)
        boundaries.append(c)

        half = min(c, cap) * math.sqrt(t_k)
        new_nodes = np.linspace(-half, half, grid_points)
        if k == 0:
            new_density = stats.norm.pdf(new_nodes, scale=math.sqrt(t_k))
        else:
            kernel = stats.norm.pdf((new_nodes[:, None] - nodes[None, :]) / step) / step
            new_density = kernel @ (weights * density)
        nodes, density = new_nodes, new_density
        weights = _simpson_weights(grid_points, new_nodes[1] - new_nodes[0])

    return GroupSequentialDesign(
        info_fractions=tuple(float(x) for x in t),
        boundaries=tuple(boundaries),
        alpha=float(spent[-1]),
        spending=spending if isinstance(spending, str) else "custom",
    )
