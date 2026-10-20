"""Data validation module for causal uplift modeling datasets.

Performs rigorous statistical and schema health checks:
1. Schema & missingness verification.
2. Binary treatment and outcome label assertions.
3. Treatment-arm balance and outcome rate verification.
4. Identical feature vector audits (documented without deleting rows,
   since identical anonymized feature vectors need not represent the same individual).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class DataValidationReport:
    """Summary of data health, schema, and causal experiment properties."""

    total_rows: int
    num_features: int
    feature_names: list[str]
    missing_values_count: int
    treatment_arm_counts: dict[str, int]
    treatment_share: float
    outcome_rates_by_arm: dict[str, float]
    raw_ate: float
    duplicate_feature_count: int
    duplicate_feature_pct: float
    is_valid: bool
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def validate_uplift_data(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    treatment_col: str = "treatment",
    outcome_col: str = "visit",
    excluded_cols: Sequence[str] | None = None,
) -> DataValidationReport:
    """Validate DataFrame conforms to causal uplift modeling requirements.

    Args:
        df: Input DataFrame.
        features: Expected list of numeric feature column names. If None, defaults to ['f0'..'f11'].
        treatment_col: Name of the binary treatment assignment column.
        outcome_col: Name of the primary binary outcome column.
        excluded_cols: Columns that must NEVER be used as features (e.g., outcome, exposure).

    Returns:
        DataValidationReport containing data quality and experimental diagnostics.

    Raises:
        ValueError: If critical schema or binary label conditions are violated.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if features is None:
        features = [f"f{i}" for i in range(12)]
    if excluded_cols is None:
        excluded_cols = ["exposure", "conversion", "visit"]

    # 1. Schema check
    required_cols = list(features) + [treatment_col, outcome_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in dataset: {missing_cols}")

    # Check for excluded column leakage into feature list
    for exc in excluded_cols:
        if exc in features:
            errors.append(
                f"Data Leakage Violation: Excluded column '{exc}' cannot be used as a pre-treatment feature."
            )

    if errors:
        raise ValueError(f"Schema validation failed:\n" + "\n".join(errors))

    # 2. Missing values check
    missing_cnt = int(df[required_cols].isna().sum().sum())
    if missing_cnt > 0:
        warnings.append(f"Dataset contains {missing_cnt} missing values across required columns.")

    # 3. Binary assertions
    t_vals = set(df[treatment_col].unique())
    if not t_vals.issubset({0, 1}):
        errors.append(f"Treatment column '{treatment_col}' must be binary {0, 1}, found: {t_vals}")

    y_vals = set(df[outcome_col].unique())
    if not y_vals.issubset({0, 1}):
        errors.append(f"Outcome column '{outcome_col}' must be binary {0, 1}, found: {y_vals}")

    if errors:
        raise ValueError("Binary validation failed:\n" + "\n".join(errors))

    # 4. Treatment-arm balance and outcome rates
    n_treated = int((df[treatment_col] == 1).sum())
    n_control = int((df[treatment_col] == 0).sum())
    total = len(df)

    if n_treated == 0 or n_control == 0:
        raise ValueError(
            f"Dataset lacks observations in both arms: treated={n_treated}, control={n_control}"
        )

    t_share = n_treated / total
    y_treated = float(df.loc[df[treatment_col] == 1, outcome_col].mean())
    y_control = float(df.loc[df[treatment_col] == 0, outcome_col].mean())
    raw_ate = y_treated - y_control

    if y_treated == 0 or y_control == 0:
        warnings.append(
            f"Rare outcome warning: one of the arms has 0 positive outcomes (Y|T=1: {y_treated:.4f}, Y|T=0: {y_control:.4f})"
        )

    # 5. Duplicate feature vectors (audit only, do not delete)
    # Check duplicate rows based strictly on features X
    dup_features = int(df.duplicated(subset=list(features)).sum())
    dup_pct = float(dup_features / total * 100)
    if dup_features > 0:
        warnings.append(
            f"Detected {dup_features} ({dup_pct:.2f}%) duplicate feature vectors. "
            "Note: These are preserved because in high-volume anonymized ad logs, "
            "multiple distinct users often share identical coarse feature representations."
        )

    return DataValidationReport(
        total_rows=total,
        num_features=len(features),
        feature_names=list(features),
        missing_values_count=missing_cnt,
        treatment_arm_counts={"control": n_control, "treated": n_treated},
        treatment_share=t_share,
        outcome_rates_by_arm={"control": y_control, "treated": y_treated},
        raw_ate=raw_ate,
        duplicate_feature_count=dup_features,
        duplicate_feature_pct=dup_pct,
        is_valid=len(errors) == 0,
        warnings=warnings,
        errors=errors,
    )
