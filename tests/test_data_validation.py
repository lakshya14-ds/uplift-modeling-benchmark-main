"""Unit tests for data validation, schema integrity, and leakage-free splitting."""

import pytest
import numpy as np
import pandas as pd

from incremental_impact.data.validator import validate_uplift_data
from incremental_impact.data.synthetic import generate_synthetic_data, SyntheticDGPConfig
from incremental_impact.data.splitter import stratified_uplift_split


def test_validation_schema_success():
    """Verify validation passes on correctly formatted synthetic data."""
    df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=500, random_seed=42))
    report = validate_uplift_data(df)
    assert report.is_valid
    assert report.total_rows == 500
    assert report.num_features == 12
    assert report.missing_values_count == 0
    assert 0.8 < report.treatment_share < 0.9


def test_validation_leakage_detection():
    """Verify error is raised if excluded columns (visit, exposure) are passed as features."""
    df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=100))
    # Deliberately attempt data leakage by including 'visit' and 'exposure' as features
    leaked_features = ["f0", "f1", "visit", "exposure"]
    with pytest.raises(ValueError, match="Data Leakage Violation"):
        validate_uplift_data(df, features=leaked_features)


def test_validation_non_binary_rejection():
    """Verify error is raised if treatment or outcome is non-binary."""
    df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=100))
    df_invalid = df.copy()
    df_invalid.loc[0, "treatment"] = 2  # Non-binary value
    with pytest.raises(ValueError, match="Binary validation failed"):
        validate_uplift_data(df_invalid)


def test_duplicate_features_audited_not_deleted():
    """Verify duplicate feature vectors are counted and documented, not deleted."""
    df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=200))
    # Force a duplicate feature row with potentially different outcome
    df.loc[1, [f"f{i}" for i in range(12)]] = df.loc[0, [f"f{i}" for i in range(12)]]
    report = validate_uplift_data(df)
    assert report.duplicate_feature_count >= 1
    assert len(df) == 200  # Rows must NOT be deleted


def test_stratified_split_ratios_and_arm_preservation():
    """Verify 60/20/20 split preserves treatment and outcome joint distributions."""
    df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=2000, random_seed=42))
    splits = stratified_uplift_split(
        df,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_seed=42,
    )

    # Check partition sizes
    assert len(splits.X_train) == 1200
    assert len(splits.X_val) == 400
    assert len(splits.X_test) == 400

    # Check treated fractions are closely preserved across splits
    t_train_share = np.mean(splits.t_train)
    t_val_share = np.mean(splits.t_val)
    t_test_share = np.mean(splits.t_test)
    assert np.isclose(t_train_share, t_val_share, atol=0.03)
    assert np.isclose(t_train_share, t_test_share, atol=0.03)

    # Check visit rates under treatment are closely preserved
    y1_train = np.mean(splits.y_train[splits.t_train == 1])
    y1_test = np.mean(splits.y_test[splits.t_test == 1])
    assert np.isclose(y1_train, y1_test, atol=0.03)


def test_preprocessing_no_leakage():
    """Verify scaler fitted strictly on train transforms val/test without refitting."""
    df, _ = generate_synthetic_data(SyntheticDGPConfig(n_samples=1000, random_seed=42))
    splits = stratified_uplift_split(df, scale_features=True, random_seed=42)

    assert splits.scaler is not None
    # Train mean should be approximately 0.0 for each feature
    train_means = splits.X_train.mean(axis=0)
    assert np.allclose(train_means, 0.0, atol=1e-5)

    # Val/test should NOT be re-centered to exactly 0 (which would indicate leakage)
    val_means = splits.X_val.mean(axis=0)
    assert not np.allclose(val_means, 0.0, atol=1e-7)
