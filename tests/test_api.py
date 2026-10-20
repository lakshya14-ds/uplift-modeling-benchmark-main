"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from incremental_impact.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    """Verify health endpoint returns healthy status and metadata."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_name" in data


def test_api_single_predict_success(client):
    """Verify single prediction with valid features."""
    sample_features = {f"f{i}": 0.5 for i in range(12)}
    res = client.post("/predict/single", json={"features": sample_features})
    assert res.status_code == 200
    data = res.json()
    assert data["total_users"] == 1
    assert len(data["predictions"]) == 1
    assert "estimated_uplift" in data["predictions"][0]
    assert "recommended_targeting" in data["predictions"][0]


def test_api_single_predict_missing_feature(client):
    """Verify 422 Unprocessable Entity when a feature is missing."""
    incomplete_features = {f"f{i}": 0.5 for i in range(11)}  # Missing f11
    res = client.post("/predict/single", json={"features": incomplete_features})
    assert res.status_code == 422


def test_api_batch_predict_success(client):
    """Verify batch prediction endpoint handles multiple records."""
    items = [{f"f{i}": float(row_idx) * 0.1 for i in range(12)} for row_idx in range(5)]
    res = client.post("/predict/batch", json={"items": items})
    assert res.status_code == 200
    data = res.json()
    assert data["total_users"] == 5
    assert len(data["predictions"]) == 5


def test_api_policy_target_endpoint(client):
    """Verify policy targeting endpoint returns binary actions under capacity limit."""
    scores = [0.05, 0.04, 0.02, 0.01, -0.01, -0.02]
    res = client.post(
        "/policy/target",
        json={"uplift_scores": scores, "capacity": 3, "strictly_positive": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_candidates"] == 6
    assert data["capacity_limit"] == 3
    assert data["targeted_count"] == 3
    assert data["policy_actions"] == [1, 1, 1, 0, 0, 0]


def test_api_simulate_budget_endpoint(client):
    """Verify budget simulation endpoint returns expected metrics and disclaimers."""
    scores = [0.04, 0.03, 0.02, 0.01, 0.005]
    payload = {
        "uplift_scores": scores,
        "budget": 2.0,
        "cost_per_target": 1.0,
        "value_per_visit": 50.0,
    }
    res = client.post("/simulate/budget", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["budget_allocated"] == 2.0
    assert data["targeted_count"] == 2
    assert data["total_spend"] == 2.0
    assert "disclaimer" in data
    assert "Scenario-based" in data["disclaimer"]


def test_api_simulate_budget_negative_value_rejected(client):
    """Verify negative budget is rejected with validation error 422."""
    payload = {
        "uplift_scores": [0.01],
        "budget": -50.0,
        "cost_per_target": 1.0,
        "value_per_visit": 10.0,
    }
    res = client.post("/simulate/budget", json=payload)
    assert res.status_code == 422
