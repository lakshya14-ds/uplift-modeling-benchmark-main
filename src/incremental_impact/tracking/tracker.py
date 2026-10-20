"""Experiment tracking module using local MLflow and embedded DuckDB.

Maintains structured, reproducible audit logs:
1. MLflow local runs (parameters, metrics, model artifacts).
2. DuckDB analytical database for fast SQL aggregation and Streamlit dashboard querying.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import duckdb
import mlflow
import pandas as pd


class ExperimentTracker:
    """Manages experiment tracking via local MLflow and DuckDB."""

    def __init__(
        self,
        mlflow_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "incremental_impact_benchmark",
        duckdb_path: str = "reports/experiment_summary.duckdb",
    ) -> None:
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.duckdb_path = Path(duckdb_path)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)

        self._init_duckdb_schema()

    def _init_duckdb_schema(self) -> None:
        with duckdb.connect(str(self.duckdb_path)) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id VARCHAR PRIMARY KEY,
                    timestamp_utc TIMESTAMP,
                    model_name VARCHAR,
                    dataset_mode VARCHAR,
                    sample_size INTEGER,
                    random_seed INTEGER,
                    base_classifier VARCHAR,
                    auuc DOUBLE,
                    qini_coefficient DOUBLE,
                    uplift_at_10 DOUBLE,
                    uplift_at_20 DOUBLE,
                    uplift_at_30 DOUBLE,
                    inc_visits_1k_top10 DOUBLE,
                    inc_visits_1k_top20 DOUBLE,
                    inc_visits_1k_top30 DOUBLE,
                    incremental_policy_value DOUBLE
                )
                """
            )

    def log_run(
        self,
        model_name: str,
        dataset_mode: str,
        sample_size: int,
        random_seed: int,
        base_classifier: str,
        metrics: dict[str, float],
        tags: dict[str, Any] | None = None,
    ) -> str:
        """Log a complete experiment run to both MLflow and DuckDB."""
        ts = datetime.now(timezone.utc)

        with mlflow.start_run(run_name=f"{model_name}_{dataset_mode}") as run:
            run_id = run.info.run_id

            # Log parameters
            params = {
                "model_name": model_name,
                "dataset_mode": dataset_mode,
                "sample_size": sample_size,
                "random_seed": random_seed,
                "base_classifier": base_classifier,
            }
            mlflow.log_params(params)

            if tags:
                mlflow.set_tags(tags)

            # Log metrics
            clean_metrics = {k: v for k, v in metrics.items() if not pd.isna(v)}
            mlflow.log_metrics(clean_metrics)

        # Log into DuckDB
        with duckdb.connect(str(self.duckdb_path)) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO experiment_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    ts,
                    model_name,
                    dataset_mode,
                    sample_size,
                    random_seed,
                    base_classifier,
                    metrics.get("auuc", None),
                    metrics.get("qini_coefficient", None),
                    metrics.get("uplift_at_10", None),
                    metrics.get("uplift_at_20", None),
                    metrics.get("uplift_at_30", None),
                    metrics.get("inc_visits_1k_top10", None),
                    metrics.get("inc_visits_1k_top20", None),
                    metrics.get("inc_visits_1k_top30", None),
                    metrics.get("incremental_policy_value", None),
                ],
            )

        return run_id

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Fetch all logged runs as a pandas DataFrame from DuckDB."""
        with duckdb.connect(str(self.duckdb_path)) as con:
            return con.execute("SELECT * FROM experiment_runs ORDER BY timestamp_utc DESC").df()
