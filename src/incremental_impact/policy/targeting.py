"""Capacity-constrained targeting policy module.

Operational Objective:
Given a fixed campaign capacity constraint (e.g., target up to K users or top fraction alpha),
select users who maximize estimated incremental outcomes while avoiding non-positive effects.

Constraint Logic:
1. Sort candidate users by estimated uplift tau_hat(x) in descending order.
2. Filter strictly for positive uplift: tau_hat(x) > 0 (avoiding dormant or negative effects).
3. Select up to K users. If fewer than K users exhibit positive uplift, select only
   those with tau_hat(x) > 0 and explicitly report the unused capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class TargetingDecisionSummary:
    """Summary of a targeting policy decision."""

    total_candidates: int
    capacity_limit: int
    targeted_count: int
    targeted_fraction: float
    unused_capacity: int
    mean_selected_uplift: float
    min_selected_uplift: float
    policy_actions: np.ndarray  # Binary 0/1 indicator for each candidate


def apply_capacity_targeting(
    uplift_scores: Sequence[float] | np.ndarray,
    capacity: int | float,
    strictly_positive: bool = True,
) -> TargetingDecisionSummary:
    """Select users under capacity constraint according to ranked uplift scores.

    Args:
        uplift_scores: 1D array of estimated uplift scores tau_hat(x).
        capacity: Either an integer count of max users, or a float fraction in (0, 1].
        strictly_positive: If True, never target users with tau_hat(x) <= 0.

    Returns:
        TargetingDecisionSummary with binary action decisions and audit statistics.
    """
    scores = np.asarray(uplift_scores, dtype=np.float64).ravel()
    n = len(scores)

    if isinstance(capacity, float) and 0.0 < capacity <= 1.0:
        k_limit = int(np.floor(capacity * n))
    else:
        k_limit = int(capacity)
    k_limit = max(0, min(k_limit, n))

    # Rank by descending score
    ranked_indices = np.argsort(-scores, kind="stable")

    # Filter positive if requested
    if strictly_positive:
        eligible_mask = scores[ranked_indices] > 0.0
        candidate_indices = ranked_indices[eligible_mask]
    else:
        candidate_indices = ranked_indices

    selected_indices = candidate_indices[:k_limit]
    actions = np.zeros(n, dtype=np.int8)
    actions[selected_indices] = 1

    n_selected = len(selected_indices)
    unused_cap = k_limit - n_selected

    mean_up = float(np.mean(scores[selected_indices])) if n_selected > 0 else 0.0
    min_up = float(np.min(scores[selected_indices])) if n_selected > 0 else 0.0

    return TargetingDecisionSummary(
        total_candidates=n,
        capacity_limit=k_limit,
        targeted_count=n_selected,
        targeted_fraction=float(n_selected / n),
        unused_capacity=unused_cap,
        mean_selected_uplift=mean_up,
        min_selected_uplift=min_up,
        policy_actions=actions,
    )
