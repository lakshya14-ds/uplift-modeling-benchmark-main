"""What-If Marketing Budget and Financial Scenario Simulator.

Operational Decision Context:
Marketing teams must allocate finite budget under cost and value parameters.
This module evaluates targeting policies under explicit unit economics:
- budget: Total available campaign budget ($).
- cost_per_target: Uniform ad delivery cost per targeted customer ($).
- value_per_visit: Estimated gross commercial value per incremental website visit ($).

Net Value Criterion:
The expected incremental net profit for an individual is:
    E[Net Value_i] = tau_hat(x_i) * value_per_visit - cost_per_target
A rational economic policy targets individuals who satisfy:
    tau_hat(x_i) > (cost_per_target / value_per_visit)
subject to the total spend constraint: (count * cost_per_target) <= budget.

CRITICAL DISCLAIMER (For Interviews & Reports):
These numbers are scenario-based simulations under simplified linear economics.
They are NOT actual Criteo auction clearing prices, real impression bidding costs,
or realized corporate profits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
import numpy as np

from incremental_impact.evaluation.policy_value import compute_policy_value


@dataclass
class ScenarioSimulationResult:
    """Detailed outcomes of a budget-constrained targeting simulation."""

    strategy_name: str
    total_population: int
    targeted_count: int
    targeted_fraction: float
    budget_allocated: float
    total_spend: float
    unspent_budget: float
    break_even_uplift_threshold: float
    mean_selected_uplift: float
    estimated_incremental_visits_predicted: float
    estimated_gross_revenue_predicted: float
    estimated_net_profit_predicted: float
    predicted_roas: float

    # Evaluated on held-out test data (IPW estimator) when outcomes y and treatment are provided
    ipw_incremental_visits: float | None = None
    ipw_gross_revenue: float | None = None
    ipw_net_profit: float | None = None
    ipw_roas: float | None = None

    disclaimer: str = (
        "Scenario-based estimate under hypothetical cost/value assumptions. "
        "Not an ad-auction simulator or guaranteed commercial ROI."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_budget_scenario(
    strategy_name: str,
    scores: Sequence[float] | np.ndarray,
    budget: float = 10000.0,
    cost_per_target: float = 0.50,
    value_per_visit: float = 25.00,
    is_response_model: bool = False,
    y: Sequence | np.ndarray | None = None,
    treatment: Sequence | np.ndarray | None = None,
    propensity: float | None = None,
) -> ScenarioSimulationResult:
    """Run budget targeting simulation for a candidate strategy.

    Args:
        strategy_name: Label for the strategy.
        scores: Predicted scores (uplift for causal models; response proba for response models).
        budget: Total spend cap ($).
        cost_per_target: Cost per ad served ($).
        value_per_visit: Dollar value per incremental visit ($).
        is_response_model: If True, scores are response probabilities P(Y=1|X, T=1) rather than uplift.
        y: Optional held-out observed outcomes for IPW counterfactual validation.
        treatment: Optional held-out treatment assignments.
        propensity: Treatment assignment probability e.

    Returns:
        ScenarioSimulationResult with predicted and empirical IPW metrics.
    """
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    n = len(scores_arr)

    if budget <= 0 or cost_per_target <= 0 or value_per_visit <= 0:
        raise ValueError("budget, cost_per_target, and value_per_visit must be strictly positive")

    max_targetable = int(np.floor(budget / cost_per_target))
    max_targetable = min(max_targetable, n)

    break_even_uplift = cost_per_target / value_per_visit

    # Selection logic
    if is_response_model:
        # Response models rank by P(Y|X, T=1), ignorant of incremental effect
        ranked_indices = np.argsort(-scores_arr, kind="stable")
        selected_indices = ranked_indices[:max_targetable]
    else:
        # Uplift model targets where expected net profit > 0: tau > break_even_uplift
        ranked_indices = np.argsort(-scores_arr, kind="stable")
        profitable_mask = scores_arr[ranked_indices] > break_even_uplift
        profitable_indices = ranked_indices[profitable_mask]
        selected_indices = profitable_indices[:max_targetable]

    targeted_count = len(selected_indices)
    targeted_fraction = targeted_count / n if n > 0 else 0.0
    total_spend = targeted_count * cost_per_target
    unspent_budget = budget - total_spend

    # Predicted metrics based on model scores (note: for response models, score is not uplift)
    if is_response_model:
        mean_uplift_pred = float("nan")
        pred_inc_visits = float("nan")
        pred_gross_rev = float("nan")
        pred_net_profit = float("nan")
        pred_roas = float("nan")
    else:
        mean_uplift_pred = float(np.mean(scores_arr[selected_indices])) if targeted_count > 0 else 0.0
        pred_inc_visits = float(mean_uplift_pred * targeted_count)
        pred_gross_rev = float(pred_inc_visits * value_per_visit)
        pred_net_profit = float(pred_gross_rev - total_spend)
        pred_roas = float(pred_gross_rev / total_spend) if total_spend > 0 else 0.0

    # Held-out empirical IPW evaluation
    ipw_visits = None
    ipw_gross = None
    ipw_net = None
    ipw_roas_val = None

    if y is not None and treatment is not None:
        policy_actions = np.zeros(n, dtype=np.int8)
        policy_actions[selected_indices] = 1
        pv_res = compute_policy_value(
            y=y,
            treatment=treatment,
            policy_actions=policy_actions,
            propensity=propensity,
        )
        # Incremental policy value is per individual across the sample of size n
        # Total incremental visits across sample = incremental_policy_value * n
        ipw_visits = float(pv_res["incremental_policy_value"] * n)
        ipw_gross = float(ipw_visits * value_per_visit)
        ipw_net = float(ipw_gross - total_spend)
        ipw_roas_val = float(ipw_gross / total_spend) if total_spend > 0 else 0.0

    return ScenarioSimulationResult(
        strategy_name=strategy_name,
        total_population=n,
        targeted_count=targeted_count,
        targeted_fraction=targeted_fraction,
        budget_allocated=budget,
        total_spend=total_spend,
        unspent_budget=unspent_budget,
        break_even_uplift_threshold=break_even_uplift,
        mean_selected_uplift=mean_uplift_pred,
        estimated_incremental_visits_predicted=pred_inc_visits,
        estimated_gross_revenue_predicted=pred_gross_rev,
        estimated_net_profit_predicted=pred_net_profit,
        predicted_roas=pred_roas,
        ipw_incremental_visits=ipw_visits,
        ipw_gross_revenue=ipw_gross,
        ipw_net_profit=ipw_net,
        ipw_roas=ipw_roas_val,
    )
