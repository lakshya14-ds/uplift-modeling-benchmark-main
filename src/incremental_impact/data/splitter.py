"""Stratified splitting and leakage-free preprocessing for causal uplift modeling.

Splits data into train (60%), validation (20%), and test (20%) partitions while
preserving the joint distribution of treatment assignment and binary outcome.
Ensures that all pre-processing (scaling, imputation, feature transforms) is fitted
strictly on the training partition to prevent data leakage into validation/test sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class UpliftDataSplits:
    """Container for train, validation, and test datasets with causal metadata."""

    X_train: pd.DataFrame
    t_train: np.ndarray
    y_train: np.ndarray

    X_val: pd.DataFrame
    t_val: np.ndarray
    y_val: np.ndarray

    X_test: pd.DataFrame
    t_test: np.ndarray
    y_test: np.ndarray

    scaler: StandardScaler | None
    feature_names: list[str]

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


def stratified_uplift_split(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    treatment_col: str = "treatment",
    outcome_col: str = "visit",
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_seed: int = 42,
    scale_features: bool = False,
) -> UpliftDataSplits:
    """Split dataset into 60/20/20 partitions stratified jointly by (treatment, outcome).

    Args:
        df: Input DataFrame.
        features: Pre-treatment feature column names. If None, uses ['f0'..'f11'].
        treatment_col: Treatment assignment column.
        outcome_col: Primary outcome column.
        train_ratio: Proportion for training (default 0.60).
        val_ratio: Proportion for validation (default 0.20).
        test_ratio: Proportion for test (default 0.20).
        random_seed: Seed for reproducible partitioning.
        scale_features: Whether to fit a StandardScaler on train and transform val/test.

    Returns:
        UpliftDataSplits containing X, t, y arrays for each partition.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got: {total_ratio}")

    if features is None:
        features = [f"f{i}" for i in range(12)]

    rng = np.random.default_rng(random_seed)
    n = len(df)

    # 4 joint strata: (T=0, Y=0), (T=0, Y=1), (T=1, Y=0), (T=1, Y=1)
    strata = df[treatment_col].to_numpy().astype(int) * 2 + df[outcome_col].to_numpy().astype(int)

    train_indices = []
    val_indices = []
    test_indices = []

    for s_val in np.unique(strata):
        idx = np.flatnonzero(strata == s_val)
        shuffled = rng.permutation(idx)
        n_stratum = len(shuffled)

        n_train = int(np.round(n_stratum * train_ratio))
        n_val = int(np.round(n_stratum * val_ratio))
        # Assign remainder to test to ensure exact conservation
        n_test = n_stratum - n_train - n_val

        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train : n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val :])

    train_idx = np.array(train_indices)
    val_idx = np.array(val_indices)
    test_idx = np.array(test_indices)

    # Shuffle within partitions
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[list(features)].copy()
    X_val = val_df[list(features)].copy()
    X_test = test_df[list(features)].copy()

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train[list(features)] = scaler.fit_transform(X_train[list(features)])
        X_val[list(features)] = scaler.transform(X_val[list(features)])
        X_test[list(features)] = scaler.transform(X_test[list(features)])

    return UpliftDataSplits(
        X_train=X_train,
        t_train=train_df[treatment_col].to_numpy().astype(int),
        y_train=train_df[outcome_col].to_numpy().astype(int),
        X_val=X_val,
        t_val=val_df[treatment_col].to_numpy().astype(int),
        y_val=val_df[outcome_col].to_numpy().astype(int),
        X_test=X_test,
        t_test=test_df[treatment_col].to_numpy().astype(int),
        y_test=test_df[outcome_col].to_numpy().astype(int),
        scaler=scaler,
        feature_names=list(features),
    )
