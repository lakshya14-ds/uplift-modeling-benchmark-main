"""Baseline models for comparative benchmarking against causal uplift estimators.

1. RandomTargetingBaseline: Ranks users uniformly at random.
2. ResponseModelBaseline: Conventional response classifier trained on treated users P(Y=1 | X, T=1).
   Ranks by conversion likelihood, which erroneously prioritizes 'Always-Takers'.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from incremental_impact.models.base import ArrayLike, BaseUpliftLearner, as_2d_numpy


class RandomTargetingBaseline(BaseUpliftLearner):
    """Random targeting baseline.

    Assigns random ranks to users. Across repeated random targeting, the expected
    uplift is equal to the sample Average Treatment Effect (ATE), and the expected
    Qini / AUUC metrics are strictly zero.
    """

    def __init__(self, random_seed: int = 42) -> None:
        super().__init__(name="RandomTargetingBaseline")
        self.random_seed = random_seed

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> RandomTargetingBaseline:
        # Random targeting requires no training parameters
        self._validate_inputs(x, treatment, y)
        self.is_fitted_ = True
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        xv = as_2d_numpy(x)
        rng = np.random.default_rng(self.random_seed)
        # Return random uniform scores in [0, 1]
        return rng.uniform(0.0, 1.0, size=xv.shape[0])


class ResponseModelBaseline(BaseUpliftLearner):
    """Conventional response model baseline.

    Trained solely on treated observations (T=1) to estimate P(Y=1 | X, T=1).
    Ranks users by their estimated likelihood of visiting when targeted.

    METHODOLOGICAL WARNING:
    This model predicts RESPONSE, NOT UPLIFT. It fails to distinguish between:
    - Persuadables: Y(1) = 1 and Y(0) = 0 (positive uplift)
    - Always-Takers: Y(1) = 1 and Y(0) = 1 (zero incremental uplift, wasted spend)
    In commercial advertising, response models often select high-intent users
    who would have visited anyway, producing high raw conversion rates but low
    or zero incremental ROI.
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        model_type: str = "lightgbm",
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=f"ResponseModelBaseline({model_type})")
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

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> ResponseModelBaseline:
        xv, tv, yv = self._validate_inputs(x, treatment, y)
        self.model_ = self._init_default_model()

        # Fit exclusively on the treated arm (T=1)
        treated_mask = tv == 1
        self.model_.fit(xv[treated_mask], yv[treated_mask])
        self.is_fitted_ = True
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        """Returns predicted probability P(Y=1 | X, T=1) as proxy ranking score."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_uplift")
        xv = as_2d_numpy(x)
        proba = self.model_.predict_proba(xv)
        classes = list(self.model_.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return proba[:, idx]
        return np.zeros(xv.shape[0])
