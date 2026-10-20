"""Learner tests: interface, ATE recovery on synthetic data, propensity sanity."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from uplift_bench.learners import SLearner, TLearner, XLearner
from uplift_bench.synthetic import FEATURES

# Fast injectable base estimators used throughout the tests.
FAST_KW = {"base_estimator": LogisticRegression(max_iter=1000)}
LEARNERS = {
    "S": lambda: SLearner(**FAST_KW),
    "T": lambda: TLearner(**FAST_KW),
    "X": lambda: XLearner(effect_estimator=Ridge(alpha=1.0), **FAST_KW),
}


def _xty(synth):
    df, truth = synth
    return df[FEATURES].to_numpy(), df["treatment"].to_numpy(), df["visit"].to_numpy(), truth


@pytest.mark.parametrize("name", list(LEARNERS))
def test_common_interface(synth, name):
    x, t, y, _ = _xty(synth)
    learner = LEARNERS[name]()
    assert learner.fit(x, t, y) is learner
    uplift = learner.predict_uplift(x[:100])
    assert uplift.shape == (100,)
    assert np.isfinite(uplift).all()


@pytest.mark.parametrize("name", list(LEARNERS))
def test_ate_recovery_on_synthetic(synth, name):
    """Numeric validation: mean predicted uplift matches the known true ATE."""
    x, t, y, truth = _xty(synth)
    true_ate = truth["tau"].mean()
    learner = LEARNERS[name]()
    est_ate = learner.fit(x, t, y).predict_uplift(x).mean()
    assert est_ate == pytest.approx(true_ate, abs=0.01)


@pytest.mark.parametrize("name", ["T", "X"])
def test_uplift_correlates_with_truth(synth, name):
    """Predicted per-row uplift ranks rows like the true effect (T and X)."""
    x, t, y, truth = _xty(synth)
    learner = LEARNERS[name]()
    uplift = learner.fit(x, t, y).predict_uplift(x)
    corr = np.corrcoef(uplift, truth["tau"])[0, 1]
    assert corr > 0.5


def test_s_learner_heterogeneity_limited_by_base_model(synth):
    """An additive base model gives the S-learner little effect heterogeneity.

    With logistic regression the treatment enters as one additive logit term,
    so predicted uplift varies only through the baseline risk. The correlation
    with the true effect stays positive but well below the T/X learners; this
    is a property of the base model, not a bug (see notebook 01).
    """
    x, t, y, truth = _xty(synth)
    uplift = LEARNERS["S"]().fit(x, t, y).predict_uplift(x)
    corr = np.corrcoef(uplift, truth["tau"])[0, 1]
    assert 0.0 < corr < 0.5


def test_default_base_estimator_is_hgb(synth):
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, t, y, _ = _xty(synth)
    learner = TLearner().fit(x[:2000], t[:2000], y[:2000])
    assert isinstance(learner.model_treated_, HistGradientBoostingClassifier)


def test_injected_estimator_is_cloned_not_shared(synth):
    x, t, y, _ = _xty(synth)
    base = LogisticRegression(max_iter=1000)
    learner = TLearner(base_estimator=base).fit(x[:2000], t[:2000], y[:2000])
    assert learner.model_treated_ is not base
    assert learner.model_treated_ is not learner.model_control_


def test_x_learner_constant_propensity(synth):
    x, t, y, _ = _xty(synth)
    learner = XLearner(effect_estimator=Ridge(), propensity="constant", **FAST_KW)
    learner.fit(x[:5000], t[:5000], y[:5000])
    e = learner.predict_propensity(x[:100])
    assert np.allclose(e, t[:5000].mean())


def test_x_learner_logistic_propensity(synth):
    x, t, y, _ = _xty(synth)
    learner = XLearner(effect_estimator=Ridge(), propensity="logistic", n_folds=3, **FAST_KW)
    learner.fit(x[:5000], t[:5000], y[:5000])
    e = learner.predict_propensity(x[:1000])
    assert ((e >= 0.01) & (e <= 0.99)).all()
    # assignment is randomized, so the fitted propensity stays near the rate
    assert abs(e.mean() - t[:5000].mean()) < 0.05


def test_validation_errors(synth):
    x, t, y, _ = _xty(synth)
    with pytest.raises(ValueError):
        SLearner(**FAST_KW).fit(x, np.full(len(x), 2), y)  # non-binary treatment
    with pytest.raises(ValueError):
        SLearner(**FAST_KW).fit(x, np.zeros(len(x)), y)  # single arm
    with pytest.raises(ValueError):
        SLearner(**FAST_KW).fit(x[:10], t[:9], y[:10])  # length mismatch
    with pytest.raises(ValueError):
        XLearner(propensity="oracle").fit(x[:100], t[:100], y[:100])  # unknown option
