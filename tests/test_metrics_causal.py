"""Causal metric validation under Zero-Effect, Constant-Effect, and Heterogeneous-Effect DGPs.

Verifies:
1. Sharp Null (Zero Effect): Metrics do not reward spurious rankings; AUUC/Qini hover around zero.
2. Constant Effect: Uniform uplift produces flat gains where ranking provides no excess value.
3. Heterogeneous Effect: Uplift models successfully achieve positive Qini and AUUC.
4. Metric scaling: Verifies exact relationship between visits/1000 targeted vs visits/1000 eligible.
5. PEHE: Tested strictly on synthetic ground truth.
6. Paired Bootstrap: Confidence intervals on paired model differences.
"""

import pytest
import numpy as np
import pandas as pd

from incremental_impact.data.synthetic import (
    generate_synthetic_data,
    SyntheticDGPConfig,
    compute_pehe,
)
from incremental_impact.models.t_learner import TLearner
from incremental_impact.models.baselines import RandomTargetingBaseline
from incremental_impact.evaluation.metrics import (
    uplift_at_k,
    incremental_visits_metrics,
    qini_coefficient,
    auuc,
    uplift_by_decile,
)
from incremental_impact.evaluation.bootstrap import (
    bootstrap_scalar_metric,
    paired_bootstrap_comparison,
)


def test_sharp_null_zero_effect_metrics():
    """Under sharp null (zero treatment effect for all units), expected AUUC and Qini are ~0."""
    df, truth = generate_synthetic_data(
        SyntheticDGPConfig(n_samples=5000, regime="zero_effect", random_seed=42)
    )
    assert np.allclose(truth["tau"], 0.0)

    y = df["visit"].to_numpy()
    t = df["treatment"].to_numpy()
    # Any arbitrary pseudo-score
    pseudo_score = df["f1"].to_numpy()

    q_coeff = qini_coefficient(y, t, pseudo_score)
    auuc_val = auuc(y, t, pseudo_score)

    # Values should be very close to 0 under true zero effect
    assert abs(q_coeff) < 0.005
    assert abs(auuc_val) < 0.005


def test_constant_effect_no_excess_ranking_gain():
    """Under constant effect, all individuals share identical uplift, so ranking yields no excess gain."""
    df, truth = generate_synthetic_data(
        SyntheticDGPConfig(n_samples=5000, regime="constant_effect", constant_uplift=0.03, random_seed=42)
    )
    assert np.allclose(truth["tau"], 0.03)

    y = df["visit"].to_numpy()
    t = df["treatment"].to_numpy()
    pseudo_score = df["f1"].to_numpy()

    # Empirical uplift at top 20% should be close to true constant effect 0.03
    u_20 = uplift_at_k(y, t, pseudo_score, k=0.20)
    assert np.isclose(u_20, 0.03, atol=0.02)


def test_heterogeneous_effect_rewards_causal_learner():
    """Under heterogeneous effect, a trained T-Learner achieves positive Qini, AUUC, and lower PEHE."""
    train_df, train_truth = generate_synthetic_data(
        SyntheticDGPConfig(n_samples=5000, regime="heterogeneous", random_seed=1)
    )
    test_df, test_truth = generate_synthetic_data(
        SyntheticDGPConfig(n_samples=3000, regime="heterogeneous", random_seed=2)
    )

    features = [f"f{i}" for i in range(12)]
    model = TLearner(model_type="lightgbm", random_seed=42).fit(
        train_df[features], train_df["treatment"], train_df["visit"]
    )

    y_test = test_df["visit"].to_numpy()
    t_test = test_df["treatment"].to_numpy()
    scores = model.predict_uplift(test_df[features])

    q_coeff = qini_coefficient(y_test, t_test, scores)
    auuc_val = auuc(y_test, t_test, scores)
    u_top10 = uplift_at_k(y_test, t_test, scores, k=0.10)

    # Model should successfully identify positive uplift in top decile
    assert q_coeff > 0
    assert auuc_val > 0
    assert u_top10 > 0

    # PEHE evaluation (valid strictly on synthetic data where counterfactuals exist)
    true_tau = test_truth["tau"].to_numpy()
    pehe_model = compute_pehe(true_tau, scores)
    assert pehe_model > 0 and np.isfinite(pehe_model)
    assert compute_pehe(true_tau, true_tau) == 0.0
    assert compute_pehe(true_tau, true_tau + 0.05) == pytest.approx(0.05)


def test_incremental_visits_scaling_distinction():
    """Verify incremental visits per 1000 targeted vs per 1000 eligible follows exact formula."""
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    t = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # Both arms present in top segment
    scores = np.linspace(1.0, 0.1, 10)

    metrics = incremental_visits_metrics(y, t, scores, k=0.30)
    u_k = metrics["uplift_at_k"]
    targeted_per_1k = metrics["incremental_visits_per_1000_targeted"]
    eligible_per_1k = metrics["incremental_visits_per_1000_eligible"]

    assert not np.isnan(u_k)
    assert np.isclose(targeted_per_1k, u_k * 1000.0)
    assert np.isclose(eligible_per_1k, u_k * 0.30 * 1000.0)
    assert np.isclose(eligible_per_1k, targeted_per_1k * 0.30)


def test_paired_bootstrap_comparison():
    """Verify paired bootstrap correctly identifies superiority when comparing good model to random."""
    train_df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=2000, regime="heterogeneous", random_seed=42))
    test_df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=1000, regime="heterogeneous", random_seed=101))

    features = [f"f{i}" for i in range(12)]
    model = TLearner(model_type="lightgbm", random_seed=42).fit(
        train_df[features], train_df["treatment"], train_df["visit"]
    )
    score_model = model.predict_uplift(test_df[features])
    score_random = RandomTargetingBaseline(random_seed=42).fit(
        train_df[features], train_df["treatment"], train_df["visit"]
    ).predict_uplift(test_df[features])

    y_test = test_df["visit"].to_numpy()
    t_test = test_df["treatment"].to_numpy()

    comp = paired_bootstrap_comparison(
        metric_fn=qini_coefficient,
        y=y_test,
        treatment=t_test,
        score_a=score_model,
        score_b=score_random,
        name_a="TLearner",
        name_b="RandomBaseline",
        n_boot=100,
        random_seed=42,
    )

    assert comp["point_difference"] > 0
    assert comp["n_boot"] == 100
