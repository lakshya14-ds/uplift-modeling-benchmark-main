"""Metric correctness on hand-computable toy cases plus bootstrap behavior."""

import numpy as np
import pytest

from uplift_bench.metrics import (
    auuc,
    bootstrap_ci,
    bootstrap_curve_band,
    cumulative_gain_curve,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
    uplift_by_decile,
)

# Toy case used across several tests. Scores rank rows 1..4 (descending).
#   rank 1: control, y=0     rank 2: treated, y=1
#   rank 3: treated, y=0     rank 4: control, y=0
TOY_Y = np.array([0, 1, 0, 0])
TOY_T = np.array([0, 1, 1, 0])
TOY_S = np.array([4.0, 3.0, 2.0, 1.0])
TOY_GRID = [0.25, 0.5, 0.75, 1.0]


def test_uplift_at_k_hand_computed():
    # top 50% = ranks 1-2: treated mean 1.0 (one row), control mean 0.0
    assert uplift_at_k(TOY_Y, TOY_T, TOY_S, k=0.5) == pytest.approx(1.0)
    # full population: treated mean 0.5, control mean 0.0
    assert uplift_at_k(TOY_Y, TOY_T, TOY_S, k=1.0) == pytest.approx(0.5)
    # top 25% = rank 1 only: control only, no treated rows -> nan
    assert np.isnan(uplift_at_k(TOY_Y, TOY_T, TOY_S, k=0.25))


def test_qini_curve_hand_computed():
    # q(phi) = (Yt - Yc * nt/nc) / n with n = 4:
    #   phi=0.25: (0 - 0*0/1)/4 = 0
    #   phi=0.50: (1 - 0*1/1)/4 = 0.25
    #   phi=0.75: (1 - 0*2/1)/4 = 0.25
    #   phi=1.00: (1 - 0*2/2)/4 = 0.25
    curve = qini_curve(TOY_Y, TOY_T, TOY_S, grid=TOY_GRID)
    np.testing.assert_allclose(curve["qini"], [0.0, 0.25, 0.25, 0.25])


def test_qini_coefficient_hand_computed():
    # diagonal = phi * 0.25; trapezoid area of (q - diagonal) over the grid
    # = 0.25 * (0.03125 + 0.09375 + 0.0625) = 0.0390625
    assert qini_coefficient(TOY_Y, TOY_T, TOY_S, grid=TOY_GRID) == pytest.approx(0.0390625)


def test_cumulative_gain_hand_computed():
    # g(phi) = (mean_yt - mean_yc) * phi:
    #   phi=0.25: nt=0 -> nan is avoided only when both arms present; here nan
    #   phi=0.50: (1/1 - 0/1) * 0.5  = 0.5
    #   phi=0.75: (1/2 - 0/1) * 0.75 = 0.375
    #   phi=1.00: (1/2 - 0/2) * 1.0  = 0.5
    curve = cumulative_gain_curve(TOY_Y, TOY_T, TOY_S, grid=TOY_GRID)
    assert np.isnan(curve["gain"].iloc[0])
    np.testing.assert_allclose(curve["gain"].iloc[1:], [0.5, 0.375, 0.5])


def _random_experiment(n=20_000, seed=11):
    """Outcomes independent of the score: any ranking is uninformative."""
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < 0.85).astype(int)
    y = (rng.random(n) < 0.05).astype(int)
    score = rng.standard_normal(n)
    return y, t, score


def test_random_score_qini_near_zero():
    y, t, score = _random_experiment()
    q = qini_coefficient(y, t, score)
    assert abs(q) < 0.002
    ci = bootstrap_ci(qini_coefficient, y, t, score, n_boot=200, seed=1)
    assert ci["lo"] < 0 < ci["hi"]


def test_random_score_auuc_near_zero():
    y, t, score = _random_experiment(seed=12)
    assert abs(auuc(y, t, score)) < 0.002


def test_perfect_ranking_beats_random():
    """Scores equal to the true effect must produce a clearly positive Qini."""
    rng = np.random.default_rng(5)
    n = 20_000
    t = (rng.random(n) < 0.85).astype(int)
    tau = np.where(rng.random(n) < 0.5, 0.10, 0.0)  # half the rows respond
    p = 0.05 + tau * t
    y = (rng.random(n) < p).astype(int)
    q_true = qini_coefficient(y, t, tau + 1e-9 * rng.standard_normal(n))
    q_rand = qini_coefficient(y, t, rng.standard_normal(n))
    assert q_true > 0.005
    assert q_true > q_rand + 0.005


def test_bootstrap_ci_reproducible_and_ordered():
    y, t, score = _random_experiment(n=5_000, seed=3)
    a = bootstrap_ci(qini_coefficient, y, t, score, n_boot=100, seed=42)
    b = bootstrap_ci(qini_coefficient, y, t, score, n_boot=100, seed=42)
    c = bootstrap_ci(qini_coefficient, y, t, score, n_boot=100, seed=43)
    assert a == b
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])
    assert a["lo"] <= a["hi"]
    assert a["n_boot"] == 100


def test_bootstrap_curve_band_shape_and_order():
    y, t, score = _random_experiment(n=5_000, seed=4)
    band = bootstrap_curve_band(qini_curve, y, t, score, n_boot=50, seed=0)
    assert list(band.columns) == ["fraction", "value", "lo", "hi"]
    assert (band["lo"] <= band["hi"]).all()


def test_uplift_by_decile_totals_and_ordering():
    y, t, score = _random_experiment(n=10_000, seed=6)
    table = uplift_by_decile(y, t, score, n_bins=10)
    assert table["n"].sum() == 10_000
    assert len(table) == 10
    # deciles are ordered by descending score
    assert table["predicted"].is_monotonic_decreasing


def test_input_validation():
    with pytest.raises(ValueError):
        uplift_at_k([1, 0], [1, 0], [0.1, 0.2], k=0.0)
    with pytest.raises(ValueError):
        uplift_at_k([1, 0, 1], [1, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        bootstrap_ci(qini_coefficient, [1, 0], [1, 0], [0.1, 0.2], n_boot=0)
