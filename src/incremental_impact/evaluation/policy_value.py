"""Policy Value and Incremental Policy Value Evaluation via Inverse Propensity Weighting (IPW).

For any candidate targeting policy pi(X) in {0, 1}:
The expected policy value V(pi) = E[Y(pi(X))] is estimated using Horvitz-Thompson IPW:
    V_hat(pi) = (1 / N) * sum_{i=1}^N [ ( I(T_i = 1, pi(X_i) = 1) / e_i ) * Y_i
                                       + ( I(T_i = 0, pi(X_i) = 0) / (1 - e_i) ) * Y_i ]

Incremental Policy Value over Status Quo (Treating Nobody, pi_0 = 0):
    Delta_V_hat(pi) = V_hat(pi) - V_hat(pi_0)
                    = (1 / N) * sum_{i: pi(X_i)=1} [ (T_i * Y_i / e_i) - ((1 - T_i) * Y_i / (1 - e_i)) ]

Properties:
- Unbiased under unconfoundedness and overlap (guaranteed in randomized experiments).
- Uses known or empirical assignment probability e(X).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_policy_value(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    policy_actions: Sequence | np.ndarray,
    propensity: float | Sequence | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute IPW policy value and incremental policy value versus treating nobody.

    Args:
        y: Observed binary outcomes (N,).
        treatment: Observed treatment assignments (N,).
        policy_actions: Binary targeting decisions pi(X) in {0, 1} (N,).
        propensity: Treatment assignment probability e(X). If None, uses sample treatment rate.

    Returns:
        dict with 'policy_value', 'treat_nobody_value', 'incremental_policy_value',
        'policy_se', and 'targeted_fraction'.
    """
    yv = np.asarray(y, dtype=np.float64).ravel()
    tv = np.asarray(treatment, dtype=np.int8).ravel()
    piv = np.asarray(policy_actions, dtype=np.int8).ravel()
    n = len(yv)

    if not (n == len(tv) == len(piv)):
        raise ValueError("y, treatment, and policy_actions must have identical lengths")

    if propensity is None:
        e = np.full(n, np.mean(tv))
    elif isinstance(propensity, (int, float)):
        e = np.full(n, float(propensity))
    else:
        e = np.asarray(propensity, dtype=np.float64).ravel()

    # Clip propensity away from 0 and 1 for numerical stability
    e = np.clip(e, 0.01, 0.99)

    # Horvitz-Thompson summands for V(pi)
    term_treated = (tv == 1) & (piv == 1)
    term_control = (tv == 0) & (piv == 0)
    w_treated = np.where(term_treated, yv / e, 0.0)
    w_control = np.where(term_control, yv / (1.0 - e), 0.0)
    policy_individual_values = w_treated + w_control
    policy_value = float(np.mean(policy_individual_values))

    # Baseline: Treat Nobody (pi_0 = 0 for everyone)
    nobody_individual_values = np.where(tv == 0, yv / (1.0 - e), 0.0)
    treat_nobody_value = float(np.mean(nobody_individual_values))

    # Incremental policy value
    incremental_value = policy_value - treat_nobody_value

    # Analytical asymptotic standard error of incremental policy value
    diff_individual = policy_individual_values - nobody_individual_values
    se_incremental = float(np.std(diff_individual, ddof=1) / np.sqrt(n))

    return {
        "policy_value": policy_value,
        "treat_nobody_value": treat_nobody_value,
        "incremental_policy_value": incremental_value,
        "incremental_policy_se": se_incremental,
        "targeted_count": int(np.sum(piv == 1)),
        "targeted_fraction": float(np.mean(piv == 1)),
    }
