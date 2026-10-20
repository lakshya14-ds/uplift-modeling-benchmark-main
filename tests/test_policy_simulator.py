"""Unit tests for capacity-constrained targeting and budget simulator."""

import pytest
import numpy as np

from incremental_impact.policy.targeting import apply_capacity_targeting
from incremental_impact.policy.simulator import simulate_budget_scenario


def test_capacity_targeting_positive_filter_and_unused_capacity():
    """Verify capacity targeting targets positive uplift only and reports unused capacity."""
    # 5 positive, 5 negative scores
    scores = np.array([0.05, 0.04, 0.03, 0.02, 0.01, -0.01, -0.02, -0.03, -0.04, -0.05])
    # Request capacity of 8
    result = apply_capacity_targeting(scores, capacity=8, strictly_positive=True)

    # Should only target the 5 positive scores!
    assert result.targeted_count == 5
    assert result.capacity_limit == 8
    assert result.unused_capacity == 3  # 8 - 5 = 3
    assert result.min_selected_uplift == 0.01
    assert result.policy_actions.sum() == 5


def test_budget_simulator_spend_cap_and_break_even():
    """Verify simulator respects budget and targets only economically profitable users."""
    scores = np.array([0.05, 0.04, 0.02, 0.01, 0.005, -0.01])
    # cost per target = $1.00, value per visit = $50.00 -> break-even uplift = 1.0 / 50.0 = 0.02
    # Users strictly profitable: scores > 0.02 -> [0.05, 0.04] (count = 2)
    # Budget = $5.00 -> max targetable = 5 users
    result = simulate_budget_scenario(
        strategy_name="UpliftPolicy",
        scores=scores,
        budget=5.0,
        cost_per_target=1.0,
        value_per_visit=50.0,
        is_response_model=False,
    )

    assert result.break_even_uplift_threshold == 0.02
    assert result.targeted_count == 2
    assert result.total_spend == 2.0
    assert result.unspent_budget == 3.0
    assert result.estimated_incremental_visits_predicted == pytest.approx(0.045 * 2)
    assert result.estimated_net_profit_predicted > 0


def test_budget_simulator_invalid_inputs():
    """Verify simulator raises error on negative or zero parameters."""
    with pytest.raises(ValueError):
        simulate_budget_scenario(
            strategy_name="Invalid",
            scores=[0.01],
            budget=-100.0,
        )
