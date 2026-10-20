"""DuckDB analytical query script for summarizing experiment runs."""

from pathlib import Path
import duckdb
import pandas as pd

DB_PATH = Path("reports/experiment_summary.duckdb")


def analyze_runs():
    if not DB_PATH.is_file():
        print(f"Database not found at {DB_PATH}. Run scripts/run_benchmark.py first.")
        return

    with duckdb.connect(str(DB_PATH)) as con:
        print("=== DuckDB Summary of Benchmark Runs ===")
        df_summary = con.execute(
            """
            SELECT 
                model_name,
                base_classifier,
                COUNT(run_id) AS num_runs,
                ROUND(AVG(auuc), 6) AS avg_auuc,
                ROUND(AVG(qini_coefficient), 6) AS avg_qini,
                ROUND(AVG(uplift_at_10) * 100, 3) AS avg_uplift_top10_pct,
                ROUND(AVG(inc_visits_1k_top10), 1) AS avg_inc_visits_per_1k_targeted,
                ROUND(AVG(incremental_policy_value), 5) AS avg_policy_value
            FROM experiment_runs
            GROUP BY model_name, base_classifier
            ORDER BY avg_auuc DESC
            """
        ).df()
        print(df_summary.to_string(index=False))


if __name__ == "__main__":
    analyze_runs()
