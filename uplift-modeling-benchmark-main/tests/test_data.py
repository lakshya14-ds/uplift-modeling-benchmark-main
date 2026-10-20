"""Data path resolution, deterministic subsampling and splitting."""

import numpy as np
import pandas as pd
import pytest

from uplift_bench import data as data_mod
from uplift_bench.data import (
    REPO_ROOT,
    criteo_path,
    data_dir,
    load_criteo,
    subsample,
    train_test_split_df,
)
from uplift_bench.synthetic import generate


def test_data_dir_default_without_env(monkeypatch):
    monkeypatch.delenv("UPLIFT_DATA_DIR", raising=False)
    assert data_dir() == REPO_ROOT / "data"


def test_data_dir_relative_env_resolves_from_repo_root(monkeypatch):
    monkeypatch.setenv("UPLIFT_DATA_DIR", "./elsewhere")
    assert data_dir() == REPO_ROOT / "elsewhere"


def test_data_dir_absolute_env(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLIFT_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    assert criteo_path() == tmp_path / "raw" / data_mod.CRITEO_FILENAME


def test_load_criteo_missing_file_error(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLIFT_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_criteo()


def test_load_criteo_reads_and_types(tmp_path):
    df, _ = generate(500, seed=1)
    path = tmp_path / "criteo.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    loaded = load_criteo(path)
    assert list(loaded.columns) == list(df.columns)
    assert loaded["f0"].dtype == np.float32
    assert loaded["treatment"].dtype == np.int8


def test_subsample_deterministic_and_stratified():
    df, _ = generate(20_000, seed=2)
    a = subsample(df, 5_000, seed=42)
    b = subsample(df, 5_000, seed=42)
    c = subsample(df, 5_000, seed=43)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert len(a) == 5_000
    assert abs(a["treatment"].mean() - df["treatment"].mean()) < 0.001
    with pytest.raises(ValueError):
        subsample(df, 0)
    with pytest.raises(ValueError):
        subsample(df, len(df) + 1)


def test_train_test_split_stratified():
    df, _ = generate(20_000, seed=3)
    train, test = train_test_split_df(df, test_size=0.5, seed=42)
    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - 0.5) < 0.01
    for part in (train, test):
        assert abs(part["treatment"].mean() - df["treatment"].mean()) < 0.005
        assert abs(part["visit"].mean() - df["visit"].mean()) < 0.005
    # seeded: identical on repeat
    train_b, _ = train_test_split_df(df, test_size=0.5, seed=42)
    pd.testing.assert_frame_equal(train, train_b)
    with pytest.raises(ValueError):
        train_test_split_df(df, test_size=1.5)
