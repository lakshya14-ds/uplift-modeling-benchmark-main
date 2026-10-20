"""S-, T- and X-learner implementations with a common fit/predict interface.

All three meta-learners estimate the conditional average treatment effect
(CATE) ``tau(x) = E[Y | X = x, T = 1] - E[Y | X = x, T = 0]`` for a binary
outcome and a binary treatment, using any scikit-learn classifier as the
base estimator:

- **S-learner**: one model on ``[X, T]``; the uplift is the difference of the
  predicted probabilities with the treatment column forced to 1 and to 0.
- **T-learner**: two models, one per treatment arm; the uplift is the
  difference of their predictions.
- **X-learner** (Kunzel et al. 2019): starts from the T-learner arm models,
  imputes individual treatment effects for each arm, regresses them on ``X``,
  and blends the two CATE estimates with propensity weights
  ``tau(x) = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)``.

The default base estimator is ``HistGradientBoostingClassifier``, which
handles the sample sizes of the Criteo dataset without preprocessing.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ArrayLike = np.ndarray


def _as_2d_float(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}")
    return arr


def _as_binary(name: str, v: ArrayLike, n: int) -> np.ndarray:
    arr = np.asarray(v).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"{name} has length {arr.shape[0]}, expected {n}")
    values = np.unique(arr)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must be binary 0/1, got values {values}")
    return arr.astype(int)


def _positive_proba(model: ClassifierMixin, x: np.ndarray) -> np.ndarray:
    """Probability of class 1, robust to a fit on a single-class subset."""
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    if 1 in classes:
        return proba[:, int(np.flatnonzero(classes == 1)[0])]
    return np.zeros(x.shape[0])


def _default_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=0)


def _default_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=0)


class BaseUpliftLearner(BaseEstimator):
    """Common interface: ``fit(X, treatment, y)`` then ``predict_uplift(X)``."""

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> BaseUpliftLearner:
        raise NotImplementedError

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        """Predicted uplift ``P(Y=1 | X, T=1) - P(Y=1 | X, T=0)`` per row."""
        raise NotImplementedError

    def _validate(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike):
        xv = _as_2d_float(x)
        tv = _as_binary("treatment", treatment, xv.shape[0])
        yv = _as_binary("y", y, xv.shape[0])
        if tv.min() == tv.max():
            raise ValueError("treatment must contain both arms (0 and 1)")
        return xv, tv, yv


class SLearner(BaseUpliftLearner):
    """Single-model learner: fit one classifier on features plus the treatment flag.

    Args:
        base_estimator: scikit-learn classifier with ``predict_proba``;
            cloned before fitting. Defaults to ``HistGradientBoostingClassifier``.
    """

    def __init__(self, base_estimator: ClassifierMixin | None = None) -> None:
        self.base_estimator = base_estimator

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> SLearner:
        xv, tv, yv = self._validate(x, treatment, y)
        base = self.base_estimator if self.base_estimator is not None else _default_classifier()
        self.model_ = clone(base)
        self.model_.fit(np.column_stack([xv, tv]), yv)
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        xv = _as_2d_float(x)
        ones = np.ones((xv.shape[0], 1))
        p1 = _positive_proba(self.model_, np.column_stack([xv, ones]))
        p0 = _positive_proba(self.model_, np.column_stack([xv, 0 * ones]))
        return p1 - p0


class TLearner(BaseUpliftLearner):
    """Two-model learner: fit one classifier per treatment arm.

    Args:
        base_estimator: scikit-learn classifier with ``predict_proba``;
            cloned once per arm. Defaults to ``HistGradientBoostingClassifier``.
    """

    def __init__(self, base_estimator: ClassifierMixin | None = None) -> None:
        self.base_estimator = base_estimator

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> TLearner:
        xv, tv, yv = self._validate(x, treatment, y)
        base = self.base_estimator if self.base_estimator is not None else _default_classifier()
        self.model_treated_ = clone(base)
        self.model_control_ = clone(base)
        self.model_treated_.fit(xv[tv == 1], yv[tv == 1])
        self.model_control_.fit(xv[tv == 0], yv[tv == 0])
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        xv = _as_2d_float(x)
        return _positive_proba(self.model_treated_, xv) - _positive_proba(self.model_control_, xv)


class XLearner(BaseUpliftLearner):
    """X-learner with propensity-weighted blending of the two CATE estimates.

    Stage 1 fits the T-learner arm models ``mu_0`` and ``mu_1``. Stage 2
    imputes individual effects, ``d_1 = y - mu_0(x)`` on the treated rows and
    ``d_0 = mu_1(x) - y`` on the control rows, and regresses each on ``X``.
    The final estimate is ``e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)``, so the
    estimate for the minority arm gets the larger weight where assignment is
    unbalanced.

    Propensity choices:

    - ``"constant"`` (default): the empirical treatment rate. Correct for a
      randomized assignment such as the Criteo experiment, where the true
      propensity does not depend on ``x``; estimating a constant avoids the
      extra variance of a fitted model.
    - ``"logistic"``: cross-fitted logistic regression
      (``cross_val_predict`` with stratified folds), for use on data where
      assignment may depend on the features. Cross-fitting keeps each row's
      propensity out-of-fold.

    Args:
        base_estimator: classifier for the stage-1 outcome models; cloned per
            arm. Defaults to ``HistGradientBoostingClassifier``.
        effect_estimator: regressor for the stage-2 imputed-effect models;
            cloned per arm. Defaults to ``HistGradientBoostingRegressor``.
        propensity: ``"constant"`` or ``"logistic"`` (see above).
        n_folds: folds for the cross-fitted logistic propensity.
        random_state: seed for the cross-fitting fold split.
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        effect_estimator: RegressorMixin | None = None,
        propensity: Literal["constant", "logistic"] = "constant",
        n_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        self.base_estimator = base_estimator
        self.effect_estimator = effect_estimator
        self.propensity = propensity
        self.n_folds = n_folds
        self.random_state = random_state

    def _fit_propensity(self, xv: np.ndarray, tv: np.ndarray) -> np.ndarray:
        if self.propensity == "constant":
            self.propensity_model_ = None
            self.propensity_rate_ = float(tv.mean())
            return np.full(xv.shape[0], self.propensity_rate_)
        if self.propensity == "logistic":
            model = LogisticRegression(max_iter=1000)
            folds = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            e_hat = cross_val_predict(model, xv, tv, cv=folds, method="predict_proba")[:, 1]
            self.propensity_model_ = clone(model).fit(xv, tv)
            self.propensity_rate_ = float(tv.mean())
            return np.clip(e_hat, 0.01, 0.99)
        raise ValueError(f"unknown propensity option: {self.propensity!r}")

    def predict_propensity(self, x: ArrayLike) -> np.ndarray:
        """Propensity ``e(x)`` used for blending (constant or fitted logistic)."""
        xv = _as_2d_float(x)
        if self.propensity_model_ is None:
            return np.full(xv.shape[0], self.propensity_rate_)
        return np.clip(self.propensity_model_.predict_proba(xv)[:, 1], 0.01, 0.99)

    def fit(self, x: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> XLearner:
        xv, tv, yv = self._validate(x, treatment, y)
        base = self.base_estimator if self.base_estimator is not None else _default_classifier()
        effect = (
            self.effect_estimator if self.effect_estimator is not None else _default_regressor()
        )

        treated, control = tv == 1, tv == 0
        self.model_treated_ = clone(base).fit(xv[treated], yv[treated])
        self.model_control_ = clone(base).fit(xv[control], yv[control])

        d_treated = yv[treated] - _positive_proba(self.model_control_, xv[treated])
        d_control = _positive_proba(self.model_treated_, xv[control]) - yv[control]
        self.effect_treated_ = clone(effect).fit(xv[treated], d_treated)
        self.effect_control_ = clone(effect).fit(xv[control], d_control)

        self._fit_propensity(xv, tv)
        return self

    def predict_uplift(self, x: ArrayLike) -> np.ndarray:
        xv = _as_2d_float(x)
        tau_treated = self.effect_treated_.predict(xv)
        tau_control = self.effect_control_.predict(xv)
        e = self.predict_propensity(xv)
        return e * tau_control + (1.0 - e) * tau_treated
