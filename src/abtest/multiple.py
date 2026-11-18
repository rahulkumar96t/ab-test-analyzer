"""Multiple-testing corrections returning adjusted p-values.

* Bonferroni and Holm control the family-wise error rate (any false positive).
* Benjamini-Hochberg controls the false discovery rate (expected share of false positives among
  the rejections) for independent or positively dependent tests.

Reject hypothesis i when ``adjusted[i] < alpha``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Correction = Literal["none", "bonferroni", "holm", "bh"]
CORRECTIONS: tuple[Correction, ...] = ("none", "bonferroni", "holm", "bh")


def adjust_pvalues(p_values: ArrayLike, method: Correction = "holm") -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    if m == 0 or method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(1.0, p * m)

    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "holm":
        # step-down: (m - i) * p_(i), made monotone non-decreasing
        stepped = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method == "bh":
        # step-up: m / i * p_(i), made monotone from the largest p-value downwards
        stepped = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
    else:
        raise ValueError(f"unknown correction {method!r}; choose from {CORRECTIONS}")
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, stepped)
    return adjusted
