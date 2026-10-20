"""Tests for the synthetic generator (schema, rates, determinism)."""

import numpy as np
import pandas as pd

from uplift_bench.data import CRITEO_COLUMNS
from uplift_bench.synthetic import generate, true_uplift_from_features


def test_schema_matches_criteo(synth):
    df, truth = synth
    assert list(df.columns) == CRITEO_COLUMNS
    assert len(df) == len(truth) == 40_000
    for col in ["treatment", "conversion", "visit", "exposure"]:
        assert set(np.unique(df[col])) <= {0, 1}


def test_treatment_rate_close_to_085(synth):
    df, _ = synth
    assert abs(df["treatment"].mean() - 0.85) < 0.01


def test_outcome_rates_realistic(synth):
    df, _ = synth
    visit_rate = df["visit"].mean()
    assert 0.02 < visit_rate < 0.12
    assert df["conversion"].mean() < visit_rate
    # conversion is a sub-event of visit
    assert (df["conversion"] <= df["visit"]).all()
    # exposure only occurs in the treated arm
    assert (df.loc[df["treatment"] == 0, "exposure"] == 0).all()


def test_true_ate_positive_and_bounded(synth):
    _, truth = synth
    ate = truth["tau"].mean()
    assert 0 < ate < 0.04
    assert (truth["tau"] >= 0).all()
    assert truth["p1"].between(0, 1).all() and truth["p0"].between(0, 1).all()


def test_difference_in_means_matches_true_ate(synth):
    """The randomized design makes diff-in-means an unbiased ATE estimate."""
    df, truth = synth
    dim = (
        df.loc[df["treatment"] == 1, "visit"].mean() - df.loc[df["treatment"] == 0, "visit"].mean()
    )
    assert abs(dim - truth["tau"].mean()) < 0.006


def test_deterministic_by_seed():
    df_a, truth_a = generate(1000, seed=3)
    df_b, truth_b = generate(1000, seed=3)
    df_c, _ = generate(1000, seed=4)
    pd.testing.assert_frame_equal(df_a, df_b)
    pd.testing.assert_frame_equal(truth_a, truth_b)
    assert not df_a.equals(df_c)


def test_true_uplift_formula(synth):
    df, truth = synth
    recomputed = true_uplift_from_features(df)
    # tau equals the formula wherever p0 + tau needed no clipping
    unclipped = truth["p1"] < 1.0
    np.testing.assert_allclose(recomputed[unclipped], truth.loc[unclipped, "tau"], atol=1e-12)
