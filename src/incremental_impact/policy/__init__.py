"""Targeting policy and marketing budget simulation modules."""

from incremental_impact.policy.targeting import (
    apply_capacity_targeting,
    TargetingDecisionSummary,
)
from incremental_impact.policy.simulator import (
    simulate_budget_scenario,
    ScenarioSimulationResult,
)

__all__ = [
    "apply_capacity_targeting",
    "TargetingDecisionSummary",
    "simulate_budget_scenario",
    "ScenarioSimulationResult",
]
