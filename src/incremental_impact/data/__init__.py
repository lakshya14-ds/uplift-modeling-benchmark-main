"""Data ingestion, validation, synthetic generation, and splitting."""

from incremental_impact.data.validator import validate_uplift_data, DataValidationReport
from incremental_impact.data.synthetic import generate_synthetic_data, SyntheticDGPConfig
from incremental_impact.data.splitter import stratified_uplift_split
from incremental_impact.data.loader import load_or_sample_criteo

__all__ = [
    "validate_uplift_data",
    "DataValidationReport",
    "generate_synthetic_data",
    "SyntheticDGPConfig",
    "stratified_uplift_split",
    "load_or_sample_criteo",
]
