"""Pydantic schemas for the IncrementalImpact FastAPI serving layer."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


class FeaturesDict(BaseModel):
    """Features f0..f11 for a single user."""

    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float
    f8: float
    f9: float
    f10: float
    f11: float


class SinglePredictRequest(BaseModel):
    features: FeaturesDict


class BatchPredictRequest(BaseModel):
    items: list[FeaturesDict] = Field(
        ..., min_length=1, max_length=10000, description="Batch of users up to 10,000"
    )


class UpliftPredictionItem(BaseModel):
    user_index: int
    estimated_uplift: float
    recommended_targeting: bool = Field(
        ..., description="True if estimated uplift is strictly positive"
    )


class PredictionResponse(BaseModel):
    model_name: str
    model_type: str
    predictions: list[UpliftPredictionItem]
    mean_estimated_uplift: float
    total_users: int


class TargetingPolicyRequest(BaseModel):
    uplift_scores: list[float] = Field(..., min_length=1)
    capacity: float = Field(..., gt=0, description="Max users (count or fraction <= 1.0)")
    strictly_positive: bool = True


class TargetingPolicyResponse(BaseModel):
    total_candidates: int
    capacity_limit: int
    targeted_count: int
    targeted_fraction: float
    unused_capacity: int
    mean_selected_uplift: float
    policy_actions: list[int]


class BudgetSimulationRequest(BaseModel):
    uplift_scores: list[float] = Field(..., min_length=1)
    budget: float = Field(..., gt=0, description="Total budget in dollars")
    cost_per_target: float = Field(..., gt=0, description="Uniform cost per targeted user ($)")
    value_per_visit: float = Field(..., gt=0, description="Value per incremental visit ($)")


class BudgetSimulationResponse(BaseModel):
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
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str
