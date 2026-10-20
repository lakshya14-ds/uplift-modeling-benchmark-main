"""S-Learner (Single Model Meta-Learner) for Uplift Modeling.

Formulation:
Fits a single classification model mu(X, T) on the concatenated matrix [X, T] to predict outcome Y.
CATE estimate:
    tau_S(x) = mu(x, 1) - mu(x, 0)
    = P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0)

Methodological Trade-offs:
- Advantage: Simple, uses all training data in a single joint estimator.
- Limitation (Regularization Bias / Treatment Shrinkage): If the outcome Y is strongly
  driven by main feature effects in X, standard regularized models (L2 penalty in logistic
  regression or tree splits in GBDTs) often split primarily on X and penalize or ignore the
  single binary treatment indicator T. Consequently, tau_S(x) is frequently biased towards zero.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from incremental_impact.models.base import ArrayLike, BaseUpliftLearner, as_2d_numpy


class SLearner(BaseUpliftLearner):
    """S-Learner for Conditional Average Treatment Effect (CATE) estimation."""

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        model_type: str = "lightgbm",
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=f"SLearner({model_type})")
        self.model_type = model_type
        self.random_seed = random_seed
        self.base_estimator = base_estimator

    def _init_default_model(self) -> ClassifierMixin:
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        if self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=self.random_seed)
        else:
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_seed,
                verbose=-1,
            )

    def _extract_positive_proba(self, model: ClassifierMixin, x_augmented: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(x_augmented)
        classes = list(model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return proba[:, idx]
        return np.zeros(x_augmented.shape[0])

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> SLearner:
        xv, tv, yv = self._validate_inputs(x, treatment, y)
        self.model_ = self._init_default_model()

        # Concatenate features X and treatment indicator T as an additional feature
        x_augmented = np.column_stack([xv, tv])
        self.model_.fit(x_augmented, yv)
        self.is_fitted_ = True
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        """Compute tau_S(x) = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_uplift")
        xv = as_2d_numpy(x)
        n = xv.shape[0]

        # Potential outcome under treatment: T forced to 1
        x_treat = np.column_stack([xv, np.ones((n, 1))])
        p1 = self._extract_positive_proba(self.model_, x_treat)

        # Potential outcome under control: T forced to 0
        x_ctrl = np.column_stack([xv, np.zeros((n, 1))])
        p0 = self._extract_positive_proba(self.model_, x_ctrl)

        return p1 - p0
