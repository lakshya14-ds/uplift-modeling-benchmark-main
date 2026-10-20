"""Base interface and abstractions for causal uplift learners."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.DataFrame | pd.Series | list


def as_2d_numpy(x: ArrayLike) -> np.ndarray:
    """Convert input features to a 2D float64 NumPy array."""
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Feature matrix X must be 2-dimensional, got shape {arr.shape}")
    return arr


def as_binary_1d(name: str, vec: ArrayLike, expected_len: int) -> np.ndarray:
    """Validate and convert treatment or outcome to a 1D binary {0, 1} int8 array."""
    if isinstance(vec, (pd.Series, pd.DataFrame)):
        arr = vec.to_numpy().ravel()
    else:
        arr = np.asarray(vec).ravel()

    if arr.shape[0] != expected_len:
        raise ValueError(f"{name} length {arr.shape[0]} does not match feature rows {expected_len}")

    unique_vals = set(np.unique(arr))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"{name} must contain only binary values in {{0, 1}}, got: {unique_vals}")
    return arr.astype(np.int8)


class BaseUpliftLearner(ABC):
    """Abstract base class for all uplift estimators.

    Enforces uniform API:
        fit(X, treatment, y) -> BaseUpliftLearner
        predict_uplift(X) -> np.ndarray
    """

    def __init__(self, name: str = "BaseUpliftLearner") -> None:
        self.name = name
        self.is_fitted_: bool = False

    def _validate_inputs(
        self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate shapes and binary domain of causal inputs."""
        xv = as_2d_numpy(x)
        tv = as_binary_1d("treatment", treatment, xv.shape[0])
        yv = as_binary_1d("y (outcome)", y, xv.shape[0])

        if np.all(tv == 0) or np.all(tv == 1):
            raise ValueError("Treatment must contain observations in both arms (0 and 1)")

        return xv, tv, yv

    @abstractmethod
    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> BaseUpliftLearner:
        """Fit the uplift learner on training data.

        Args:
            x: Pre-treatment feature matrix (N, d).
            treatment: Binary treatment assignment vector (N,).
            y: Observed binary outcome vector (N,).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        """Estimate the Conditional Average Treatment Effect (CATE) tau(x) = E[Y|X=x, T=1] - E[Y|X=x, T=0].

        Args:
            x: Feature matrix (M, d).

        Returns:
            1D array of estimated uplift scores tau(x) of shape (M,).
        """
        raise NotImplementedError

    def save(self, filepath: str | Path) -> None:
        """Serialize fitted model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> BaseUpliftLearner:
        """Load serialized model from disk."""
        path = Path(filepath)
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, BaseUpliftLearner):
            raise TypeError(f"Loaded object is not a BaseUpliftLearner: {type(model)}")
        return model
