"""Synthetic uplift data with a known heterogeneous treatment effect.

The generator mimics the schema of the Criteo uplift dataset (12 numeric
features ``f0``-``f11``, binary ``treatment``, ``visit``, ``conversion`` and
``exposure``) but the treatment effect is known exactly per row, so metric
and learner code can be validated against ground truth before any real data
is touched.

Design:

- assignment is randomized with a constant rate (default 0.85, like Criteo);
- the control outcome probability depends on ``f0`` and ``f2`` and averages
  a few percent, comparable to the Criteo ``visit`` rate;
- the true uplift ``tau(x)`` is non-negative and heterogeneous, driven by
  ``f1`` and ``f3``; rows with low ``f1`` have essentially zero effect;
- ``conversion`` is a rare sub-event of ``visit`` and carries no separate
  effect, mirroring why the benchmark models ``visit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

FEATURES = [f"f{i}" for i in range(12)]
COLUMNS = [*FEATURES, "treatment", "conversion", "visit", "exposure"]


def true_uplift_from_features(x: pd.DataFrame, max_uplift: float = 0.04) -> np.ndarray:
    """True uplift ``tau(x) = max_uplift * expit(2.0 * f1 + 0.8 * f3)`` per row."""
    return max_uplift * expit(2.0 * x["f1"].to_numpy() + 0.8 * x["f3"].to_numpy())


def generate(
    n: int,
    seed: int = 0,
    treatment_rate: float = 0.85,
    base_rate_logit: float = -3.2,
    max_uplift: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic uplift dataset with known ground truth.

    Args:
        n: number of rows.
        seed: seed for the random generator; the output is deterministic.
        treatment_rate: probability of assignment to the treated arm.
        base_rate_logit: intercept of the control outcome model; the default
            gives a control visit rate of roughly 4 to 5 percent.
        max_uplift: upper bound of the true per-row uplift.

    Returns:
        ``(df, truth)`` where ``df`` has the Criteo columns and ``truth`` has
        per-row ``p0``, ``p1`` and ``tau`` (the known treatment effect).
        ``truth["tau"].mean()`` is the true ATE of the sample.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if not 0 < treatment_rate < 1:
        raise ValueError("treatment_rate must be in (0, 1)")
    rng = np.random.default_rng(seed)

    x = pd.DataFrame(rng.standard_normal((n, len(FEATURES))), columns=FEATURES)
    treatment = (rng.random(n) < treatment_rate).astype(np.int8)

    p0 = expit(base_rate_logit + 0.8 * x["f0"].to_numpy() - 0.5 * x["f2"].to_numpy())
    tau = true_uplift_from_features(x, max_uplift=max_uplift)
    p1 = np.clip(p0 + tau, 0.0, 1.0)

    p_assigned = np.where(treatment == 1, p1, p0)
    visit = (rng.random(n) < p_assigned).astype(np.int8)
    # Conversion is a rare sub-event of visit with no additional effect.
    conversion = (visit & (rng.random(n) < 0.06)).astype(np.int8)
    # Exposure marks treated users who would actually see the ad.
    exposure = (treatment & (rng.random(n) < 0.3)).astype(np.int8)

    df = x.copy()
    df["treatment"] = treatment
    df["conversion"] = conversion
    df["visit"] = visit
    df["exposure"] = exposure
    truth = pd.DataFrame({"p0": p0, "p1": p1, "tau": p1 - p0})
    return df[COLUMNS], truth
