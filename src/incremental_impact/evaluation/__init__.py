"""Evaluation metrics, policy value estimators, and bootstrap uncertainty modules."""

from incremental_impact.evaluation.metrics import (
    uplift_at_k,
    incremental_visits_metrics,
    qini_curve,
    qini_coefficient,
    cumulative_gain_curve,
    auuc,
    uplift_by_decile,
)
from incremental_impact.evaluation.policy_value import compute_policy_value
from incremental_impact.evaluation.bootstrap import (
    bootstrap_scalar_metric,
    paired_bootstrap_comparison,
)

__all__ = [
    "uplift_at_k",
    "incremental_visits_metrics",
    "qini_curve",
    "qini_coefficient",
    "cumulative_gain_curve",
    "auuc",
    "uplift_by_decile",
    "compute_policy_value",
    "bootstrap_scalar_metric",
    "paired_bootstrap_comparison",
]
