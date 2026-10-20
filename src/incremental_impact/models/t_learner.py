"""T-Learner (Two Model Meta-Learner) for Uplift Modeling.

Formulation:
Fits two separate models for the two treatment arms:
1. mu_1(X) trained on treated users (T=1) predicting Y.
2. mu_0(X) trained on control users (T=0) predicting Y.

CATE estimate:
    tau_T(x) = mu_1(x) - mu_0(x)
    = P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0)

Methodological Trade-offs:
- Advantage: Completely avoids the S-Learner's regularization shrinkage bias on the treatment effect.
  Feature interactions with treatment are learned completely independently.
- Limitation (Arm Imbalance Variance): In digital advertising experiments like Criteo,
  assignment is heavily imbalanced (~85% treated, ~15% control). The control model mu_0(X)
  is trained on significantly fewer observations and may suffer from higher variance or overfitting,
  introducing noise into the difference tau_T(x) = mu_1(x) - mu_0(x).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from incremental_impact.models.base import ArrayLike, BaseUpliftLearner, as_2d_numpy


class TLearner(BaseUpliftLearner):
    """T-Learner for Conditional Average Treatment Effect (CATE) estimation."""

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        model_type: str = "lightgbm",
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=f"TLearner({model_type})")
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

    def _extract_positive_proba(self, model: ClassifierMixin, x: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(x)
        classes = list(model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return proba[:, idx]
        return np.zeros(x.shape[0])

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> TLearner:
        xv, tv, yv = self._validate_inputs(x, treatment, y)

        self.model_treated_ = self._init_default_model()
        self.model_control_ = self._init_default_model()

        # Arm masks
        treated = tv == 1
        control = tv == 0

        self.model_treated_.fit(xv[treated], yv[treated])
        self.model_control_.fit(xv[control], yv[control])
        self.is_fitted_ = True
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        """Compute tau_T(x) = mu_1(x) - mu_0(x)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_uplift")
        xv = as_2d_numpy(x)
        p1 = self._extract_positive_proba(self.model_treated_, xv)
        p0 = self._extract_positive_proba(self.model_control_, xv)
        return p1 - p0
