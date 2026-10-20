"""Smoke tests for the shared plot helpers (Agg backend, no display)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uplift_bench.plotting import PALETTE, plot_curves_with_bands, plot_uplift_by_decile, set_style


def test_set_style_applies_white_background():
    set_style()
    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["font.size"] >= 11
    assert len(PALETTE) <= 6


def test_plot_curves_with_bands_smoke():
    band = pd.DataFrame(
        {
            "fraction": np.linspace(0.1, 1, 10),
            "value": np.linspace(0, 0.01, 10),
            "lo": np.linspace(-0.001, 0.008, 10),
            "hi": np.linspace(0.001, 0.012, 10),
        }
    )
    ax = plot_curves_with_bands({"model": band}, title="t", ylabel="qini", diagonal_from="model")
    assert ax.get_title() == "t"
    assert len(ax.lines) == 2  # curve + diagonal
    plt.close("all")


def test_plot_uplift_by_decile_smoke():
    table = pd.DataFrame(
        {
            "decile": np.arange(1, 11),
            "n": 100,
            "predicted": np.linspace(0.02, 0.0, 10),
            "observed": np.linspace(0.025, -0.005, 10),
            "se": 0.005,
        }
    )
    ax = plot_uplift_by_decile(table, title="calibration")
    assert ax.get_title() == "calibration"
    plt.close("all")
