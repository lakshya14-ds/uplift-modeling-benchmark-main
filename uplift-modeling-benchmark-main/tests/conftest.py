"""Shared fixtures: a small synthetic dataset with known ground truth."""

import pytest

from uplift_bench.synthetic import generate


@pytest.fixture(scope="session")
def synth():
    """Synthetic dataset (40k rows) with per-row true uplift, seed fixed."""
    df, truth = generate(40_000, seed=7)
    return df, truth
