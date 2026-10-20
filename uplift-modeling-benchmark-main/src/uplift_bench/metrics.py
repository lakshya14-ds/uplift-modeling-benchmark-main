"""Uplift evaluation metrics with bootstrap confidence intervals.

All metrics take three aligned arrays: the observed binary outcome ``y``,
the binary treatment indicator ``treatment`` and a per-row uplift ``score``
(higher means the model expects a larger treatment effect). They evaluate
ranking quality only; they never look at the scale of the scores.

Conventions used here (definitions vary across papers and packages):

- ``uplift_at_k``: difference of outcome means between treated and control
  rows inside the top ``k`` fraction by score.
- ``qini_curve``: at fraction ``phi``, the incremental number of outcomes
  among the treated in the top segment, with the control sum rescaled to the
  treated group size, divided by the total sample size ``n`` so that values
  are per-individual and comparable across sample sizes:
  ``q(phi) = (Y_t(phi) - Y_c(phi) * n_t(phi) / n_c(phi)) / n``.
- ``qini_coefficient``: area between the Qini curve and the random-targeting
  diagonal ``phi * q(1)``. Zero in expectation for a random score.
- ``cumulative_gain_curve``: ``g(phi) = (ybar_t(phi) - ybar_c(phi)) * phi``,
  the expected incremental outcome per individual when the top ``phi`` is
  targeted.
- ``auuc``: area under the cumulative gain curve minus the area under its
  random-targeting diagonal, so it is also zero-centered.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

DEFAULT_GRID = np.linspace(0.05, 1.0, 20)


def _validate(y, treatment, score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yv = np.asarray(y, dtype=float).ravel()
    tv = np.asarray(treatment, dtype=int).ravel()
    sv = np.asarray(score, dtype=float).ravel()
    if not (yv.shape == tv.shape == sv.shape):
        raise ValueError("y, treatment and score must have the same length")
    if yv.shape[0] == 0:
        raise ValueError("empty input")
    return yv, tv, sv


def _sort_desc(yv: np.ndarray, tv: np.ndarray, sv: np.ndarray):
    order = np.argsort(-sv, kind="stable")
    return yv[order], tv[order]


def uplift_at_k(y, treatment, score, k: float = 0.1) -> float:
    """Uplift among the top ``k`` fraction ranked by score.

    Returns ``mean(y | t=1, top) - mean(y | t=0, top)``, or ``nan`` when the
    top segment lacks one of the arms.
    """
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    yv, tv, sv = _validate(y, treatment, score)
    y_top, t_top = _sort_desc(yv, tv, sv)
    n_top = int(np.ceil(k * yv.shape[0]))
    y_top, t_top = y_top[:n_top], t_top[:n_top]
    n_t, n_c = int(t_top.sum()), int((1 - t_top).sum())
    if n_t == 0 or n_c == 0:
        return float("nan")
    return float(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())


def qini_curve(y, treatment, score, grid: Sequence[float] | None = None) -> pd.DataFrame:
    """Qini curve on a fraction grid.

    Returns a DataFrame with columns ``fraction`` and ``qini`` where ``qini``
    is the per-individual incremental outcome defined in the module
    docstring. Segments that lack control rows yield ``nan``.
    """
    yv, tv, sv = _validate(y, treatment, score)
    fractions = np.asarray(DEFAULT_GRID if grid is None else grid, dtype=float)
    y_sorted, t_sorted = _sort_desc(yv, tv, sv)
    n = yv.shape[0]

    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1 - t_sorted))
    cum_nt = np.cumsum(t_sorted)
    cum_nc = np.cumsum(1 - t_sorted)

    idx = np.ceil(fractions * n).astype(int).clip(1, n) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(
            cum_nc[idx] > 0,
            (cum_yt[idx] - cum_yc[idx] * cum_nt[idx] / cum_nc[idx]) / n,
            np.nan,
        )
    return pd.DataFrame({"fraction": fractions, "qini": q})


def qini_coefficient(y, treatment, score, grid: Sequence[float] | None = None) -> float:
    """Area between the Qini curve and the random-targeting diagonal.

    Positive when ranking by ``score`` concentrates the treatment effect in
    the top segments; approximately zero for a random score.
    """
    curve = qini_curve(y, treatment, score, grid=grid)
    fractions = curve["fraction"].to_numpy()
    q = curve["qini"].to_numpy()
    if np.isnan(q).any():
        return float("nan")
    diagonal = fractions * q[-1]
    return float(np.trapezoid(q - diagonal, fractions))


def cumulative_gain_curve(y, treatment, score, grid: Sequence[float] | None = None) -> pd.DataFrame:
    """Cumulative gain ``g(phi) = (ybar_t(phi) - ybar_c(phi)) * phi`` on a grid."""
    yv, tv, sv = _validate(y, treatment, score)
    fractions = np.asarray(DEFAULT_GRID if grid is None else grid, dtype=float)
    y_sorted, t_sorted = _sort_desc(yv, tv, sv)
    n = yv.shape[0]

    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1 - t_sorted))
    cum_nt = np.cumsum(t_sorted)
    cum_nc = np.cumsum(1 - t_sorted)

    idx = np.ceil(fractions * n).astype(int).clip(1, n) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(
            (cum_nt[idx] > 0) & (cum_nc[idx] > 0),
            (cum_yt[idx] / cum_nt[idx] - cum_yc[idx] / cum_nc[idx]) * fractions,
            np.nan,
        )
    return pd.DataFrame({"fraction": fractions, "gain": gain})


def auuc(y, treatment, score, grid: Sequence[float] | None = None) -> float:
    """Area under the cumulative gain curve, centered on the random diagonal."""
    curve = cumulative_gain_curve(y, treatment, score, grid=grid)
    fractions = curve["fraction"].to_numpy()
    gain = curve["gain"].to_numpy()
    if np.isnan(gain).any():
        return float("nan")
    diagonal = fractions * gain[-1]
    return float(np.trapezoid(gain - diagonal, fractions))


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    y,
    treatment,
    score,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a scalar uplift metric.

    Rows are resampled with replacement within each treatment arm
    (stratified), so every resample keeps both arms at their original sizes.
    Each resample is shuffled before scoring: predicted scores can contain
    ties, and an arm-ordered resample would otherwise place control rows
    first within every tied block, which systematically biases rank metrics.

    Args:
        metric: callable ``metric(y, treatment, score) -> float``.
        n_boot: number of bootstrap resamples (reduce under ``CI=true``).
        alpha: two-sided level; the default 0.05 gives a 95 percent interval.
        seed: seed for the resampling generator.

    Returns:
        Dict with ``estimate`` (on the full sample), ``lo``, ``hi`` and
        ``n_boot``. Resamples that return ``nan`` are dropped.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    yv, tv, sv = _validate(y, treatment, score)
    rng = np.random.default_rng(seed)
    idx_by_arm = [np.flatnonzero(tv == arm) for arm in (0, 1)]

    values = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([rng.choice(idx, size=idx.shape[0]) for idx in idx_by_arm])
        rng.shuffle(take)
        values[b] = metric(yv[take], tv[take], sv[take])
    values = values[~np.isnan(values)]
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": float(metric(yv, tv, sv)),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(values.shape[0]),
    }


def bootstrap_curve_band(
    curve_fn: Callable[..., pd.DataFrame],
    y,
    treatment,
    score,
    grid: Sequence[float] | None = None,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Pointwise percentile bootstrap band for a curve metric.

    Args:
        curve_fn: ``qini_curve`` or ``cumulative_gain_curve``.

    Returns:
        DataFrame with columns ``fraction``, ``value`` (full-sample curve),
        ``lo`` and ``hi``.
    """
    yv, tv, sv = _validate(y, treatment, score)
    fractions = np.asarray(DEFAULT_GRID if grid is None else grid, dtype=float)
    rng = np.random.default_rng(seed)
    idx_by_arm = [np.flatnonzero(tv == arm) for arm in (0, 1)]

    full = curve_fn(yv, tv, sv, grid=fractions).iloc[:, 1].to_numpy()
    boots = np.empty((n_boot, fractions.shape[0]))
    for b in range(n_boot):
        take = np.concatenate([rng.choice(idx, size=idx.shape[0]) for idx in idx_by_arm])
        rng.shuffle(take)
        boots[b] = curve_fn(yv[take], tv[take], sv[take], grid=fractions).iloc[:, 1].to_numpy()
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return pd.DataFrame({"fraction": fractions, "value": full, "lo": lo, "hi": hi})


def uplift_by_decile(y, treatment, score, n_bins: int = 10) -> pd.DataFrame:
    """Observed uplift and mean predicted uplift per score decile.

    Rows are split into ``n_bins`` equal-count bins by score (bin 0 = highest
    scores). Returns one row per bin with the mean predicted uplift, the
    observed uplift (difference of outcome means) and its normal-approximation
    standard error.
    """
    yv, tv, sv = _validate(y, treatment, score)
    order = np.argsort(-sv, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        yb, tb, sb = yv[idx], tv[idx], sv[idx]
        n_t, n_c = int(tb.sum()), int((1 - tb).sum())
        if n_t == 0 or n_c == 0:
            observed, se = float("nan"), float("nan")
        else:
            p_t, p_c = yb[tb == 1].mean(), yb[tb == 0].mean()
            observed = float(p_t - p_c)
            se = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))
        rows.append(
            {
                "decile": i + 1,
                "n": int(idx.shape[0]),
                "predicted": float(sb.mean()),
                "observed": observed,
                "se": se,
            }
        )
    return pd.DataFrame(rows)
