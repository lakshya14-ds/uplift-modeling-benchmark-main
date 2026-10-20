"""Loading, deterministic subsampling and splitting of the Criteo uplift data.

Paths are configured through the ``UPLIFT_DATA_DIR`` environment variable
(read from a ``.env`` file via python-dotenv when present). The default is
``./data`` relative to the repository root, so everything works without a
``.env`` file and regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CRITEO_FILENAME = "criteo-research-uplift-v2.1.csv.gz"

FEATURES = [f"f{i}" for i in range(12)]
CRITEO_COLUMNS = [*FEATURES, "treatment", "conversion", "visit", "exposure"]


def data_dir() -> Path:
    """Resolve the data directory from ``UPLIFT_DATA_DIR`` (default ``./data``).

    Relative values are resolved against the repository root, not the current
    working directory, so notebooks behave the same from any location.
    """
    load_dotenv(REPO_ROOT / ".env")
    raw = os.environ.get("UPLIFT_DATA_DIR", "./data")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def criteo_path() -> Path:
    """Expected location of the raw Criteo file (see scripts/download_data.py)."""
    return data_dir() / "raw" / CRITEO_FILENAME


def criteo_available() -> bool:
    """Whether the raw Criteo file has been downloaded."""
    return criteo_path().is_file()


def load_criteo(path: Path | None = None) -> pd.DataFrame:
    """Load the full Criteo uplift dataset (about 13.9M rows).

    Features are read as float32 and flags as int8 to keep the frame around
    750 MB in memory.

    Raises:
        FileNotFoundError: with a pointer to the download script.
    """
    path = criteo_path() if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Criteo dataset not found at {path.name} under the data directory. "
            "Run: python scripts/download_data.py"
        )
    dtype: dict[str, type] = {c: np.float32 for c in FEATURES}
    dtype.update({c: np.int8 for c in ["treatment", "conversion", "visit", "exposure"]})
    df = pd.read_csv(path, dtype=dtype)
    missing = [c for c in CRITEO_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected schema, missing columns: {missing}")
    return df[CRITEO_COLUMNS]


def subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic subsample of ``n`` rows, stratified by ``treatment``.

    Each arm contributes a share proportional to its size (largest remainder
    on the treated arm), so the treated fraction is preserved to within one
    row. Sampling is without replacement and seeded; the same ``(n, seed)``
    always returns the same rows, in a shuffled order with a fresh index.
    """
    if not 0 < n <= len(df):
        raise ValueError(f"n must be in 1..{len(df)}")
    rng = np.random.default_rng(seed)
    t = df["treatment"].to_numpy()
    idx_treated = np.flatnonzero(t == 1)
    idx_control = np.flatnonzero(t == 0)
    n_treated = round(n * idx_treated.shape[0] / len(df))
    n_control = n - n_treated
    take = np.concatenate(
        [
            rng.choice(idx_treated, size=n_treated, replace=False),
            rng.choice(idx_control, size=n_control, replace=False),
        ]
    )
    rng.shuffle(take)
    return df.iloc[take].reset_index(drop=True)


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded row split stratified jointly by ``treatment`` and ``visit``.

    Stratifying on the treatment-outcome combination keeps both the treated
    fraction and the (rare) outcome rate per arm equal across the two parts.

    Returns:
        ``(train, test)`` with fresh integer indexes.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(seed)
    strata = df["treatment"].to_numpy() * 2 + df["visit"].to_numpy()
    test_mask = np.zeros(len(df), dtype=bool)
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        n_test = round(len(idx) * test_size)
        test_mask[rng.choice(idx, size=n_test, replace=False)] = True
    train = df.loc[~test_mask].reset_index(drop=True)
    test = df.loc[test_mask].reset_index(drop=True)
    return train, test
