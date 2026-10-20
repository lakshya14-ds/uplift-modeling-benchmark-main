"""Shared matplotlib style and plot helpers for the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

#: Categorical palette (max six colors, no plain tab10, no red-green pair).
PALETTE = ["#3b6ea5", "#c66a2c", "#4b8a5a", "#7b5aa6", "#b3473f", "#6b6b6b"]


def set_style() -> None:
    """Apply the shared plot style (call once at the top of a notebook)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 110,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )


def plot_curves_with_bands(
    bands: dict[str, pd.DataFrame],
    title: str,
    ylabel: str,
    diagonal_from: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot one or more bootstrap curve bands (from ``bootstrap_curve_band``).

    Args:
        bands: mapping from label to a DataFrame with columns ``fraction``,
            ``value``, ``lo`` and ``hi``.
        title: axes title; state the conclusion, not the plot type.
        ylabel: y-axis label including units.
        diagonal_from: label of the band whose endpoint anchors a dashed
            random-targeting diagonal, or ``None`` for no diagonal.
        ax: existing axes, or ``None`` to create a new figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, band) in enumerate(bands.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(band["fraction"], band["value"], label=label, color=color, lw=1.8)
        ax.fill_between(band["fraction"], band["lo"], band["hi"], color=color, alpha=0.18, lw=0)
    if diagonal_from is not None:
        ref = bands[diagonal_from]
        endpoint = float(ref["value"].iloc[-1])
        ax.plot([0, 1], [0, endpoint], ls="--", color="#999999", lw=1.2, label="random targeting")
    ax.set_xlabel("fraction of population targeted (by score)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def plot_uplift_by_decile(
    table: pd.DataFrame,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Calibration bar plot from ``metrics.uplift_by_decile`` output.

    Bars show the observed uplift per score decile (with normal-approximation
    error bars); dots show the mean predicted uplift of the same decile.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    x = table["decile"].to_numpy()
    ax.bar(
        x,
        table["observed"],
        yerr=1.96 * table["se"],
        color=PALETTE[0],
        alpha=0.75,
        capsize=3,
        label="observed uplift",
        error_kw={"elinewidth": 1.0, "ecolor": "#555555"},
    )
    ax.plot(x, table["predicted"], "o", color=PALETTE[1], ms=6, label="mean predicted uplift")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("score decile (1 = highest predicted uplift)")
    ax.set_ylabel("uplift (difference in outcome rate)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax
