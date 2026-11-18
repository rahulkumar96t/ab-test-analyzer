"""Static chart for the peeking simulation (PNG, no pyplot global state)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

from abtest.validation import PeekingRow, Rate

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {  # fixed categorical order: blue, orange, aqua
    "msprt": ("mSPRT (always valid)", "#2a78d6"),
    "naive": ("naive peeking", "#eb6834"),
    "gst": ("group-sequential, OBF spending", "#1baf7a"),
}


def plot_peeking(rows: Sequence[PeekingRow], path: str | Path, alpha: float = 0.05) -> Path:
    """False-positive rate against the number of looks, with 95 % Monte-Carlo intervals."""
    rows = sorted((r for r in rows if r.lift == 0), key=lambda r: r.looks)
    looks = np.array([r.looks for r in rows])
    fig = Figure(figsize=(7.2, 4.2), dpi=110, facecolor=SURFACE)
    ax = fig.add_subplot(facecolor=SURFACE)
    ax.axhline(alpha, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    mid = float(np.sqrt(looks[0] * looks[-1]))
    ax.text(mid, alpha * 0.92, f"nominal alpha {alpha:.0%}", color=MUTED, va="top", fontsize=9)
    for key, (label, color) in SERIES.items():
        rates: list[Rate] = [getattr(r, key) for r in rows]
        values = np.array([r.value for r in rates])
        bounds = np.array([r.wilson() for r in rates])
        ax.fill_between(looks, bounds[:, 0], bounds[:, 1], color=color, alpha=0.15, lw=0)
        ax.plot(looks, values, color=color, lw=2, marker="o", ms=5, label=label, zorder=3)
        ax.annotate(
            f"{values[-1]:.1%}",
            (looks[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(looks.tolist()))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(k) for k in looks])
    ax.set_xlim(looks[0] * 0.85, looks[-1] * 1.6)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("number of looks at the data (same total sample)", color=MUTED, fontsize=9)
    ax.set_ylabel("false-positive rate (A/A tests)", color=MUTED, fontsize=9)
    sims = rows[0].naive.trials
    ax.set_title(
        f"Stopping at the first significant look, {sims:,} simulated A/A tests per point",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    return out
