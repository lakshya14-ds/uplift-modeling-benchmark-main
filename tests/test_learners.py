"""Unit tests for uplift learners, baselines, deterministic inference, and serialization."""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from incremental_impact.data.synthetic import generate_synthetic_data, SyntheticDGPConfig
from incremental_impact.models.baselines import RandomTargetingBaseline, ResponseModelBaseline
from incremental_impact.models.s_learner import SLearner
from incremental_impact.models.t_learner import TLearner
from incremental_impact.models.x_learner import XLearner
from incremental_impact.models.base import BaseUpliftLearner


@pytest.fixture
def small_synthetic_data():
    df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=500, random_seed=42))
    features = [f"f{i}" for i in range(12)]
    X = df[features]
    t = df["treatment"].to_numpy()
    y = df["visit"].to_numpy()
    return X, t, y


@pytest.mark.parametrize(
    "model_cls,model_type",
    [
        (RandomTargetingBaseline, None),
        (ResponseModelBaseline, "logistic_regression"),
        (ResponseModelBaseline, "lightgbm"),
        (SLearner, "logistic_regression"),
        (SLearner, "lightgbm"),
        (TLearner, "logistic_regression"),
        (TLearner, "lightgbm"),
        (XLearner, "logistic_regression"),
        (XLearner, "lightgbm"),
    ],
)
def test_learners_fit_predict_contract(small_synthetic_data, model_cls, model_type):
    """Verify common fit(X, t, y) and predict_uplift(X) contract across all learners."""
    X, t, y = small_synthetic_data
    if model_type is None:
        model = model_cls(random_seed=42)
    else:
        model = model_cls(model_type=model_type, random_seed=42)

    model.fit(X, t, y)
    scores = model.predict_uplift(X)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(X),)
    assert not np.isnan(scores).any()


def test_deterministic_inference(small_synthetic_data):
    """Verify identical predictions are produced across repeated calls with fixed seed."""
    X, t, y = small_synthetic_data

    model1 = TLearner(model_type="lightgbm", random_seed=42).fit(X, t, y)
    scores1 = model1.predict_uplift(X)

    model2 = TLearner(model_type="lightgbm", random_seed=42).fit(X, t, y)
    scores2 = model2.predict_uplift(X)

    assert np.allclose(scores1, scores2)


def test_model_serialization(small_synthetic_data):
    """Verify save and load reproduces exact predictions."""
    X, t, y = small_synthetic_data
    model = XLearner(model_type="lightgbm", random_seed=42).fit(X, t, y)
    original_scores = model.predict_uplift(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.pkl"
        model.save(model_path)
        assert model_path.is_file()

        loaded_model = BaseUpliftLearner.load(model_path)
        loaded_scores = loaded_model.predict_uplift(X)
        assert np.allclose(original_scores, loaded_scores)


def test_learner_invalid_inputs(small_synthetic_data):
    """Verify learners raise informative errors on corrupted inputs."""
    X, t, y = small_synthetic_data
    model = SLearner()

    # Corrupt treatment: only 1 arm present
    t_single_arm = np.ones_like(t)
    with pytest.raises(ValueError, match="Treatment must contain observations in both arms"):
        model.fit(X, t_single_arm, y)

    # 1D feature array instead of 2D
    with pytest.raises(ValueError, match="Feature matrix X must be 2-dimensional"):
        model.fit(X["f0"].to_numpy(), t, y)
