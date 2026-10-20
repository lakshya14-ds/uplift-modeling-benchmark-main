"""X-Learner (Causal Two-Stage Meta-Learner with Propensity Blending).

Formulation (Künzel et al., PNAS 2019):
Specifically engineered for datasets with substantial treatment-assignment imbalance (such as Criteo's 85% treated arm):

Stage 1 (Outcome Estimation):
- Fit base classifier mu_1(X) on treated data (T=1).
- Fit base classifier mu_0(X) on control data (T=0).

Stage 2 (Counterfactual Imputation):
- For treated users: impute counterfactual control outcome mu_0(X_1) and compute imputed treatment effect:
    D_1 = Y_1 - mu_0(X_1)
- For control users: impute counterfactual treated outcome mu_1(X_0) and compute imputed treatment effect:
    D_0 = mu_1(X_0) - Y_0

Stage 3 (CATE Regression):
- Train effect regressor tau_1(X) predicting D_1 on treated users.
- Train effect regressor tau_0(X) predicting D_0 on control users.

Stage 4 (Propensity-Weighted Blending):
- Combine the two CATE models using the propensity score e(x) = P(T=1 | X=x):
    tau_X(x) = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)

Why the Weighting Scheme Works (Intuition for Interviews):
When the treated arm is very large (e.g., e = 0.85):
- The treated outcome model mu_1(X) is estimated with high precision.
- Therefore, the imputed effect D_0 = mu_1(X_0) - Y_0 for control users has very low imputation noise.
- Model tau_0(X) is trained on accurate pseudo-effects, so we give it a large weight e(x) = 0.85!
- Conversely, mu_0(X) is trained on few control samples and has higher variance, making D_1 noisier;
  hence, tau_1(X) receives a smaller weight (1 - e(x)) = 0.15.
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import lightgbm as lgb

from incremental_impact.models.base import ArrayLike, BaseUpliftLearner, as_2d_numpy


class XLearner(BaseUpliftLearner):
    """X-Learner for Conditional Average Treatment Effect (CATE) estimation."""

    def __init__(
        self,
        base_classifier: ClassifierMixin | None = None,
        effect_regressor: RegressorMixin | None = None,
        model_type: str = "lightgbm",
        propensity_type: Literal["constant", "logistic"] = "constant",
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=f"XLearner({model_type}, propensity={propensity_type})")
        self.model_type = model_type
        self.propensity_type = propensity_type
        self.random_seed = random_seed
        self.base_classifier = base_classifier
        self.effect_regressor = effect_regressor

    def _init_default_classifier(self) -> ClassifierMixin:
        if self.base_classifier is not None:
            return clone(self.base_classifier)
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

    def _init_default_regressor(self) -> RegressorMixin:
        if self.effect_regressor is not None:
            return clone(self.effect_regressor)
        if self.model_type == "logistic_regression":
            return Ridge(alpha=1.0, random_state=self.random_seed)
        else:
            return lgb.LGBMRegressor(
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

    def _fit_propensity(self, xv: np.ndarray, tv: np.ndarray) -> np.ndarray:
        self.propensity_rate_ = float(tv.mean())
        if self.propensity_type == "constant":
            self.propensity_model_ = None
            return np.full(xv.shape[0], self.propensity_rate_)
        elif self.propensity_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=self.random_seed)
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
            e_hat = cross_val_predict(model, xv, tv, cv=folds, method="predict_proba")[:, 1]
            self.propensity_model_ = clone(model).fit(xv, tv)
            return np.clip(e_hat, 0.01, 0.99)
        else:
            raise ValueError(f"Unknown propensity type: {self.propensity_type}")

    def predict_propensity(self, x: np.ndarray) -> np.ndarray:
        """Estimate propensity score e(x) = P(T=1 | X=x)."""
        if self.propensity_model_ is None:
            return np.full(x.shape[0], self.propensity_rate_)
        return np.clip(self.propensity_model_.predict_proba(x)[:, 1], 0.01, 0.99)

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> XLearner:
        xv, tv, yv = self._validate_inputs(x, treatment, y)

        treated = tv == 1
        control = tv == 0

        # Stage 1: Fit outcome classifiers mu_1 and mu_0
        self.mu_1_ = self._init_default_classifier().fit(xv[treated], yv[treated])
        self.mu_0_ = self._init_default_classifier().fit(xv[control], yv[control])

        # Stage 2: Compute imputed treatment effects
        d_1 = yv[treated] - self._extract_positive_proba(self.mu_0_, xv[treated])
        d_0 = self._extract_positive_proba(self.mu_1_, xv[control]) - yv[control]

        # Stage 3: Fit effect regressors
        self.tau_1_ = self._init_default_regressor().fit(xv[treated], d_1)
        self.tau_0_ = self._init_default_regressor().fit(xv[control], d_0)

        # Stage 4: Fit propensity model / rate
        self._fit_propensity(xv, tv)

        self.is_fitted_ = True
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        """Compute blended CATE: tau_X(x) = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_uplift")
        xv = as_2d_numpy(x)
        tau_0_pred = self.tau_0_.predict(xv)
        tau_1_pred = self.tau_1_.predict(xv)
        e = self.predict_propensity(xv)
        return e * tau_0_pred + (1.0 - e) * tau_1_pred
