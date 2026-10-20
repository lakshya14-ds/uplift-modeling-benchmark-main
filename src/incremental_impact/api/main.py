"""FastAPI service for causal uplift inference, policy recommendation, and budget simulation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from incremental_impact import __version__
from incremental_impact.api.schemas import (
    BatchPredictRequest,
    BudgetSimulationRequest,
    BudgetSimulationResponse,
    HealthResponse,
    PredictionResponse,
    SinglePredictRequest,
    TargetingPolicyRequest,
    TargetingPolicyResponse,
    UpliftPredictionItem,
)
from incremental_impact.models.base import BaseUpliftLearner
from incremental_impact.models.x_learner import XLearner
from incremental_impact.policy.simulator import simulate_budget_scenario
from incremental_impact.policy.targeting import apply_capacity_targeting

MODEL_ARTIFACT_PATH = Path("artifacts/models/best_uplift_model.pkl")

# Global model state
_state: dict[str, Any] = {
    "model": None,
    "model_name": "None",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Load serialized uplift model at startup."""
    if MODEL_ARTIFACT_PATH.is_file():
        try:
            loaded = BaseUpliftLearner.load(MODEL_ARTIFACT_PATH)
            _state["model"] = loaded
            _state["model_name"] = loaded.name
            print(f"Loaded production model artifact: {loaded.name}")
        except Exception as e:
            print(f"Warning: Failed loading {MODEL_ARTIFACT_PATH}: {e}")
    else:
        # Fallback to an untrained placeholder or lightweight default
        print(f"Artifact not found at {MODEL_ARTIFACT_PATH}. Serving in fallback state.")

    yield
    _state["model"] = None


app = FastAPI(
    title="IncrementalImpact API",
    description="Production-grade API for serving Causal Uplift Models (CATE) and Budget Simulations",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_active_model() -> BaseUpliftLearner:
    model = _state.get("model")
    if model is None:
        # Lazy check if file was generated after startup
        if MODEL_ARTIFACT_PATH.is_file():
            loaded = BaseUpliftLearner.load(MODEL_ARTIFACT_PATH)
            _state["model"] = loaded
            _state["model_name"] = loaded.name
            return loaded
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained uplift model loaded. Please run 'python scripts/run_benchmark.py' to generate artifacts.",
        )
    return model


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service status and active uplift model metadata."""
    model = _state.get("model")
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=_state.get("model_name", "None"),
        version=__version__,
    )


@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(req: SinglePredictRequest) -> PredictionResponse:
    """Estimate Conditional Average Treatment Effect (CATE) for a single user."""
    model = _get_active_model()
    feat_dict = req.features.model_dump()
    row_df = pd.DataFrame([feat_dict])

    uplift_score = float(model.predict_uplift(row_df)[0])
    item = UpliftPredictionItem(
        user_index=0,
        estimated_uplift=uplift_score,
        recommended_targeting=bool(uplift_score > 0),
    )

    return PredictionResponse(
        model_name=model.name,
        model_type=type(model).__name__,
        predictions=[item],
        mean_estimated_uplift=uplift_score,
        total_users=1,
    )


@app.post("/predict/batch", response_model=PredictionResponse)
def predict_batch(req: BatchPredictRequest) -> PredictionResponse:
    """Estimate CATE uplift for a batch of users (up to 10,000)."""
    model = _get_active_model()
    records = [item.model_dump() for item in req.items]
    batch_df = pd.DataFrame(records)

    scores = model.predict_uplift(batch_df)
    items = [
        UpliftPredictionItem(
            user_index=i,
            estimated_uplift=float(score),
            recommended_targeting=bool(score > 0),
        )
        for i, score in enumerate(scores)
    ]

    return PredictionResponse(
        model_name=model.name,
        model_type=type(model).__name__,
        predictions=items,
        mean_estimated_uplift=float(np.mean(scores)),
        total_users=len(items),
    )


@app.post("/policy/target", response_model=TargetingPolicyResponse)
def apply_targeting_policy(req: TargetingPolicyRequest) -> TargetingPolicyResponse:
    """Recommend capacity-constrained targeting actions."""
    res = apply_capacity_targeting(
        uplift_scores=req.uplift_scores,
        capacity=req.capacity,
        strictly_positive=req.strictly_positive,
    )
    return TargetingPolicyResponse(
        total_candidates=res.total_candidates,
        capacity_limit=res.capacity_limit,
        targeted_count=res.targeted_count,
        targeted_fraction=res.targeted_fraction,
        unused_capacity=res.unused_capacity,
        mean_selected_uplift=res.mean_selected_uplift,
        policy_actions=res.policy_actions.tolist(),
    )


@app.post("/simulate/budget", response_model=BudgetSimulationResponse)
def simulate_budget(req: BudgetSimulationRequest) -> BudgetSimulationResponse:
    """Simulate commercial marketing campaign under explicit budget and unit economics."""
    res = simulate_budget_scenario(
        strategy_name="UpliftPolicy",
        scores=req.uplift_scores,
        budget=req.budget,
        cost_per_target=req.cost_per_target,
        value_per_visit=req.value_per_visit,
        is_response_model=False,
    )
    return BudgetSimulationResponse(
        total_population=res.total_population,
        targeted_count=res.targeted_count,
        targeted_fraction=res.targeted_fraction,
        budget_allocated=res.budget_allocated,
        total_spend=res.total_spend,
        unspent_budget=res.unspent_budget,
        break_even_uplift_threshold=res.break_even_uplift_threshold,
        mean_selected_uplift=res.mean_selected_uplift,
        estimated_incremental_visits_predicted=res.estimated_incremental_visits_predicted,
        estimated_gross_revenue_predicted=res.estimated_gross_revenue_predicted,
        estimated_net_profit_predicted=res.estimated_net_profit_predicted,
        predicted_roas=res.predicted_roas,
        disclaimer=res.disclaimer,
    )
