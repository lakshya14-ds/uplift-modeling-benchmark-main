"""Paired bootstrap confidence intervals for model comparisons on held-out evaluation sets.

Methodological Principle:
When comparing two targeting strategies (e.g., Uplift Model vs Response Baseline),
they must be evaluated on the EXACT SAME resampled test units to isolate ranking superiority
from sample variability.

Stratified Bootstrap:
Resampling is stratified by treatment arm (sampling with replacement within T=1 and T=0),
preserving the treatment-control ratio across all B resamples.

Note on Scope of Uncertainty:
Fixed-model bootstrap confidence intervals quantify evaluation-sample uncertainty
conditional on the trained models. They do not quantify retraining variance from
different training splits/seeds (which is separately evaluated via multi-seed runs).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence
import numpy as np
import pandas as pd

from incremental_impact.evaluation.metrics import validate_metric_inputs


def bootstrap_scalar_metric(
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> dict[str, float]:
    """Compute percentile bootstrap CI for a single model's scalar uplift metric.

    Args:
        metric_fn: Callable f(y, treatment, score) -> float.
        y: Observed binary outcomes.
        treatment: Binary treatment assignments.
        score: Model predicted uplift scores.
        n_boot: Number of bootstrap resamples (default 200).
        alpha: Significance level (default 0.05 -> 95% CI).
        random_seed: Random seed for reproducibility.

    Returns:
        dict with 'estimate', 'ci_lower', 'ci_upper', 'std_error', 'n_boot'.
    """
    yv, tv, sv = validate_metric_inputs(y, treatment, score)
    rng = np.random.default_rng(random_seed)

    idx_treated = np.flatnonzero(tv == 1)
    idx_control = np.flatnonzero(tv == 0)

    boot_vals = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_treated = rng.choice(idx_treated, size=len(idx_treated), replace=True)
        sample_control = rng.choice(idx_control, size=len(idx_control), replace=True)
        take = np.concatenate([sample_treated, sample_control])
        rng.shuffle(take)

        val = metric_fn(yv[take], tv[take], sv[take])
        boot_vals[b] = val

    valid_vals = boot_vals[~np.isnan(boot_vals)]
    if len(valid_vals) == 0:
        return {
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "std_error": float("nan"),
            "n_boot": 0,
        }

    lo = np.percentile(valid_vals, 100 * (alpha / 2))
    hi = np.percentile(valid_vals, 100 * (1 - alpha / 2))
    point_est = metric_fn(yv, tv, sv)

    return {
        "estimate": float(point_est),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std_error": float(np.std(valid_vals, ddof=1)),
        "n_boot": len(valid_vals),
    }


def paired_bootstrap_comparison(
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score_a: Sequence | np.ndarray,
    score_b: Sequence | np.ndarray,
    name_a: str = "Model_A",
    name_b: str = "Model_B",
    n_boot: int = 200,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute paired bootstrap difference: Delta = Metric(A) - Metric(B).

    Both models are scored on the identical resampled holdout units to control for test-set variance.
    """
    yv, tv, sa = validate_metric_inputs(y, treatment, score_a)
    _, _, sb = validate_metric_inputs(y, treatment, score_b)

    rng = np.random.default_rng(random_seed)
    idx_treated = np.flatnonzero(tv == 1)
    idx_control = np.flatnonzero(tv == 0)

    diffs = np.empty(n_boot, dtype=float)
    vals_a = np.empty(n_boot, dtype=float)
    vals_b = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_treated = rng.choice(idx_treated, size=len(idx_treated), replace=True)
        sample_control = rng.choice(idx_control, size=len(idx_control), replace=True)
        take = np.concatenate([sample_treated, sample_control])
        rng.shuffle(take)

        yt = yv[take]
        tt = tv[take]
        sat = sa[take]
        sbt = sb[take]

        ma = metric_fn(yt, tt, sat)
        mb = metric_fn(yt, tt, sbt)
        vals_a[b] = ma
        vals_b[b] = mb
        diffs[b] = ma - mb

    mask = ~np.isnan(diffs)
    valid_diffs = diffs[mask]
    point_a = metric_fn(yv, tv, sa)
    point_b = metric_fn(yv, tv, sb)
    point_diff = point_a - point_b

    diff_lo = float(np.percentile(valid_diffs, 100 * (alpha / 2)))
    diff_hi = float(np.percentile(valid_diffs, 100 * (1 - alpha / 2)))

    # A difference is statistically significant at level alpha if 0 is outside [diff_lo, diff_hi]
    is_significant = (diff_lo > 0) or (diff_hi < 0)

    return {
        "model_a": name_a,
        "model_b": name_b,
        "metric_name": metric_fn.__name__,
        "estimate_a": float(point_a),
        "estimate_b": float(point_b),
        "point_difference": float(point_diff),
        "diff_ci_lower": diff_lo,
        "diff_ci_upper": diff_hi,
        "is_significant": bool(is_significant),
        "p_value_approx": float(2.0 * min(np.mean(valid_diffs <= 0), np.mean(valid_diffs >= 0))),
        "n_boot": len(valid_diffs),
    }
