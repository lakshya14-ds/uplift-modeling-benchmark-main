"""Synthetic Data Generating Process (DGP) for Causal Uplift Modeling.

Provides 3 controlled ground-truth regimes to rigorously test causal estimators:
1. 'heterogeneous': Non-negative heterogeneous treatment effects driven by nonlinear features.
2. 'zero_effect': Sharp null where treatment assignment has strictly zero causal effect (tau(x) = 0).
3. 'constant_effect': Homogeneous treatment effect where tau(x) = constant for all x.

Critical Note: Ground truth counterfactual probabilities (p0, p1) and individual treatment effects
(tau = p1 - p0) are available ONLY in this synthetic module. In real datasets (e.g. Criteo),
individual counterfactuals are inherently unobservable due to the Fundamental Problem of Causal Inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy.special import expit

FEATURES = [f"f{i}" for i in range(12)]
COLUMNS = [*FEATURES, "treatment", "conversion", "visit", "exposure"]


@dataclass
class SyntheticDGPConfig:
    n_samples: int = 20000
    regime: Literal["heterogeneous", "zero_effect", "constant_effect"] = "heterogeneous"
    treatment_rate: float = 0.85  # Reflects Criteo ~85% treated imbalance
    base_rate_logit: float = -3.2 # Base visit rate ~4-5%
    max_uplift: float = 0.04      # Max heterogeneous uplift ~4%
    constant_uplift: float = 0.02 # For constant effect regime
    random_seed: int = 42


def true_uplift_function(
    x: pd.DataFrame,
    regime: Literal["heterogeneous", "zero_effect", "constant_effect"] = "heterogeneous",
    max_uplift: float = 0.04,
    constant_uplift: float = 0.02,
) -> np.ndarray:
    """Compute true conditional average treatment effect tau(x) under the specified regime."""
    n = len(x)
    if regime == "zero_effect":
        return np.zeros(n, dtype=float)
    elif regime == "constant_effect":
        return np.full(n, constant_uplift, dtype=float)
    elif regime == "heterogeneous":
        # Nonlinear uplift driven by f1 and f3; users with low f1 have virtually zero uplift
        f1 = x["f1"].to_numpy()
        f3 = x["f3"].to_numpy()
        return max_uplift * expit(2.0 * f1 + 0.8 * f3)
    else:
        raise ValueError(f"Unknown synthetic regime: {regime}")


def generate_synthetic_data(
    config: SyntheticDGPConfig | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic uplift dataset with known counterfactuals and potential outcomes.

    Returns:
        tuple of (observed_df, ground_truth_df)
        - observed_df contains: [f0..f11, treatment, conversion, visit, exposure]
        - ground_truth_df contains: [p0, p1, tau, regime]
    """
    if config is None:
        config = SyntheticDGPConfig(**kwargs)

    rng = np.random.default_rng(config.random_seed)
    n = config.n_samples

    # Generate 12 continuous features
    x_matrix = rng.standard_normal((n, len(FEATURES)))
    x = pd.DataFrame(x_matrix, columns=FEATURES)

    # Randomized treatment assignment
    treatment = (rng.random(n) < config.treatment_rate).astype(np.int8)

    # Baseline outcome probability under control: P(Y=1 | X, T=0)
    # Driven primarily by f0 and f2
    p0 = expit(config.base_rate_logit + 0.8 * x["f0"].to_numpy() - 0.5 * x["f2"].to_numpy())

    # CATE: tau(x) = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)
    tau = true_uplift_function(
        x,
        regime=config.regime,
        max_uplift=config.max_uplift,
        constant_uplift=config.constant_uplift,
    )
    p1 = np.clip(p0 + tau, 0.0, 1.0)
    # Recompute exact tau after probability clipping
    tau = p1 - p0

    # Realized visit outcome based on assigned arm
    p_assigned = np.where(treatment == 1, p1, p0)
    visit = (rng.random(n) < p_assigned).astype(np.int8)

    # Secondary outcome: conversion is a rare downstream sub-event of visit
    conversion = (visit & (rng.random(n) < 0.06)).astype(np.int8)

    # Exposure indicates if ad was actually served to an eligible treated user
    exposure = (treatment & (rng.random(n) < 0.30)).astype(np.int8)

    observed_df = x.copy()
    observed_df["treatment"] = treatment
    observed_df["conversion"] = conversion
    observed_df["visit"] = visit
    observed_df["exposure"] = exposure

    ground_truth_df = pd.DataFrame(
        {
            "p0": p0,
            "p1": p1,
            "tau": tau,
            "regime": config.regime,
        }
    )

    return observed_df[COLUMNS], ground_truth_df


def compute_pehe(true_tau: np.ndarray, pred_tau: np.ndarray) -> float:
    """Compute Precision in Estimating Heterogeneous Effects (PEHE): sqrt(mean((tau_pred - tau_true)^2)).

    NOTE: PEHE requires true individual causal effects tau(x). It CANNOT be computed on real-world
    observational or experimental data (such as Criteo) because individual counterfactuals are never observed.
    """
    true_tau = np.asarray(true_tau).ravel()
    pred_tau = np.asarray(pred_tau).ravel()
    if true_tau.shape != pred_tau.shape:
        raise ValueError(f"Shape mismatch: {true_tau.shape} vs {pred_tau.shape}")
    return float(np.sqrt(np.mean((pred_tau - true_tau) ** 2)))
