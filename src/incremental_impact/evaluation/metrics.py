"""Causal Uplift Evaluation Metrics: Qini, AUUC, Uplift@k, and Decile Diagnostics.

All metrics evaluate the quality of ranking units by estimated uplift using observed
outcomes (Y) and randomized treatment assignment (T) on held-out test data.

Conventions & Mathematical Formulations:
1. Uplift@k:
   Mean difference in outcomes between treated and control units in the top k fraction:
   Uplift@k = E[Y | T=1, top k] - E[Y | T=0, top k]

2. Incremental Visits per 1,000 Targeted Users:
   Scaled expected incremental lift among those selected:
   Delta_Visits_Targeted = Uplift@k * 1000

   DISTINCTION FROM Total Eligible Population:
   Delta_Visits_Total_Eligible = Uplift@k * k * 1000

3. Qini Curve (Radcliffe, 2007):
   At fraction phi:
   Q(phi) = ( Y_t(phi) - Y_c(phi) * (n_t(phi) / n_c(phi)) ) / N
   where Y_t, Y_c are cumulative outcomes in top phi, and n_t, n_c are cumulative counts.

4. Qini Coefficient:
   The area between the Qini curve and the random-targeting diagonal line connecting
   (0, 0) to (1, Q(1)). A random ranking has expected Qini coefficient of 0.

5. Cumulative Gain Curve:
   G(phi) = ( mean(Y | T=1, top phi) - mean(Y | T=0, top phi) ) * phi

6. AUUC (Area Under Uplift / Cumulative Gain Curve):
   Area between the cumulative gain curve and the random diagonal. Zero-centered for random scores.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

DEFAULT_GRID = np.linspace(0.05, 1.0, 20)


def validate_metric_inputs(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align outcome, treatment, and predicted score vectors."""
    yv = np.asarray(y, dtype=np.float64).ravel()
    tv = np.asarray(treatment, dtype=np.int8).ravel()
    sv = np.asarray(score, dtype=np.float64).ravel()

    if not (len(yv) == len(tv) == len(sv)):
        raise ValueError(
            f"Aligned length mismatch: y={len(yv)}, treatment={len(tv)}, score={len(sv)}"
        )
    if len(yv) == 0:
        raise ValueError("Input arrays must not be empty")

    return yv, tv, sv


def sort_by_descending_score(
    yv: np.ndarray, tv: np.ndarray, sv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort inputs in descending order of predicted uplift scores using stable sort."""
    order = np.argsort(-sv, kind="stable")
    return yv[order], tv[order], sv[order]


def uplift_at_k(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    k: float = 0.10,
) -> float:
    """Compute empirical uplift (mean(Y|T=1) - mean(Y|T=0)) in the top k fraction.

    Args:
        y: Observed binary outcomes.
        treatment: Binary treatment assignments.
        score: Predicted uplift ranking scores.
        k: Fraction of users targeted (0 < k <= 1.0).

    Returns:
        Uplift float, or nan if top segment lacks treated or control units.
    """
    if not (0.0 < k <= 1.0):
        raise ValueError(f"k must be in (0, 1], got: {k}")

    yv, tv, sv = validate_metric_inputs(y, treatment, score)
    y_sort, t_sort, _ = sort_by_descending_score(yv, tv, sv)

    n_top = int(np.ceil(k * len(yv)))
    y_top = y_sort[:n_top]
    t_top = t_sort[:n_top]

    n_t = int(np.sum(t_top == 1))
    n_c = int(np.sum(t_top == 0))

    if n_t == 0 or n_c == 0:
        return float("nan")

    return float(np.mean(y_top[t_top == 1]) - np.mean(y_top[t_top == 0]))


def incremental_visits_metrics(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    k: float = 0.10,
) -> dict[str, float]:
    """Calculate incremental visits per 1,000 targeted users vs per 1,000 eligible users."""
    u_k = uplift_at_k(y, treatment, score, k=k)
    return {
        "k_fraction": float(k),
        "uplift_at_k": float(u_k),
        "incremental_visits_per_1000_targeted": float(u_k * 1000.0) if not np.isnan(u_k) else float("nan"),
        "incremental_visits_per_1000_eligible": float(u_k * k * 1000.0) if not np.isnan(u_k) else float("nan"),
    }


def qini_curve(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compute Radcliffe Qini curve over a targeting fraction grid.

    Returns DataFrame with columns:
        fraction: targeting fraction phi
        qini: normalized incremental outcomes per individual in sample
        n_targeted: number of users targeted
        n_treated: treated users in segment
        n_control: control users in segment
        y_treated: outcomes in treated segment
        y_control: outcomes in control segment
    """
    yv, tv, sv = validate_metric_inputs(y, treatment, score)
    fractions = np.asarray(DEFAULT_GRID if grid is None else grid, dtype=np.float64)
    y_sort, t_sort, _ = sort_by_descending_score(yv, tv, sv)
    n = len(yv)

    cum_yt = np.cumsum(y_sort * t_sort)
    cum_yc = np.cumsum(y_sort * (1 - t_sort))
    cum_nt = np.cumsum(t_sort)
    cum_nc = np.cumsum(1 - t_sort)

    # Convert fraction to index (1-based, capped at n)
    indices = np.ceil(fractions * n).astype(int).clip(1, n) - 1

    qini_vals = []
    for idx in indices:
        nt = cum_nt[idx]
        nc = cum_nc[idx]
        yt = cum_yt[idx]
        yc = cum_yc[idx]
        if nc > 0:
            q = (yt - yc * (nt / nc)) / n
        else:
            q = np.nan
        qini_vals.append(q)

    return pd.DataFrame(
        {
            "fraction": fractions,
            "qini": qini_vals,
            "n_targeted": indices + 1,
            "n_treated": cum_nt[indices],
            "n_control": cum_nc[indices],
            "y_treated": cum_yt[indices],
            "y_control": cum_yc[indices],
        }
    )


def _trapz(y_vals: np.ndarray, x_vals: np.ndarray) -> float:
    """Trapezoidal integration compatible with both NumPy 1.x (trapz) and 2.x (trapezoid)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_vals, x_vals))
    return float(np.trapz(y_vals, x_vals))


def qini_coefficient(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    grid: Sequence[float] | None = None,
) -> float:
    """Compute Qini coefficient: Area between Qini curve and random targeting baseline."""
    df_qini = qini_curve(y, treatment, score, grid=grid)
    fractions = df_qini["fraction"].to_numpy()
    q = df_qini["qini"].to_numpy()

    valid = ~np.isnan(q)
    if not valid.any():
        return float("nan")

    f_valid = fractions[valid]
    q_valid = q[valid]

    # Prepend origin (0, 0) for consistent trapezoidal integration
    if f_valid[0] > 0:
        f_valid = np.insert(f_valid, 0, 0.0)
        q_valid = np.insert(q_valid, 0, 0.0)

    diagonal = f_valid * q_valid[-1]
    return _trapz(q_valid - diagonal, f_valid)


def cumulative_gain_curve(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compute cumulative gain curve G(phi) = (mean(Y|T=1) - mean(Y|T=0)) * phi."""
    yv, tv, sv = validate_metric_inputs(y, treatment, score)
    fractions = np.asarray(DEFAULT_GRID if grid is None else grid, dtype=np.float64)
    y_sort, t_sort, _ = sort_by_descending_score(yv, tv, sv)
    n = len(yv)

    cum_yt = np.cumsum(y_sort * t_sort)
    cum_yc = np.cumsum(y_sort * (1 - t_sort))
    cum_nt = np.cumsum(t_sort)
    cum_nc = np.cumsum(1 - t_sort)

    indices = np.ceil(fractions * n).astype(int).clip(1, n) - 1

    gains = []
    for i, idx in enumerate(indices):
        nt = cum_nt[idx]
        nc = cum_nc[idx]
        yt = cum_yt[idx]
        yc = cum_yc[idx]
        phi = fractions[i]
        if nt > 0 and nc > 0:
            gain = (yt / nt - yc / nc) * phi
        else:
            gain = np.nan
        gains.append(gain)

    return pd.DataFrame({"fraction": fractions, "gain": gains})


def auuc(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    grid: Sequence[float] | None = None,
) -> float:
    """Compute Area Under Uplift Curve (AUUC) centered against random diagonal baseline."""
    df_gain = cumulative_gain_curve(y, treatment, score, grid=grid)
    fractions = df_gain["fraction"].to_numpy()
    gain = df_gain["gain"].to_numpy()

    valid = ~np.isnan(gain)
    if not valid.any():
        return float("nan")

    f_valid = fractions[valid]
    g_valid = gain[valid]

    if f_valid[0] > 0:
        f_valid = np.insert(f_valid, 0, 0.0)
        g_valid = np.insert(g_valid, 0, 0.0)

    diagonal = f_valid * g_valid[-1]
    return _trapz(g_valid - diagonal, f_valid)


def uplift_by_decile(
    y: Sequence | np.ndarray,
    treatment: Sequence | np.ndarray,
    score: Sequence | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute decile breakdown of predicted scores vs observed empirical uplift."""
    yv, tv, sv = validate_metric_inputs(y, treatment, score)
    order = np.argsort(-sv, kind="stable")
    bins = np.array_split(order, n_bins)

    rows = []
    for i, idx in enumerate(bins):
        yb = yv[idx]
        tb = tv[idx]
        sb = sv[idx]

        n_tot = len(idx)
        n_t = int(np.sum(tb == 1))
        n_c = int(np.sum(tb == 0))

        if n_t == 0 or n_c == 0:
            observed = float("nan")
            se = float("nan")
        else:
            p_t = np.mean(yb[tb == 1])
            p_c = np.mean(yb[tb == 0])
            observed = float(p_t - p_c)
            # Analytical normal-approximation standard error
            var_t = (p_t * (1 - p_t)) / n_t
            var_c = (p_c * (1 - p_c)) / n_c
            se = float(np.sqrt(var_t + var_c))

        rows.append(
            {
                "decile": i + 1,
                "n_users": n_tot,
                "n_treated": n_t,
                "n_control": n_c,
                "mean_predicted_uplift": float(np.mean(sb)),
                "observed_uplift": observed,
                "standard_error": se,
                "ci_lower_95": observed - 1.96 * se if not np.isnan(se) else float("nan"),
                "ci_upper_95": observed + 1.96 * se if not np.isnan(se) else float("nan"),
            }
        )

    return pd.DataFrame(rows)
