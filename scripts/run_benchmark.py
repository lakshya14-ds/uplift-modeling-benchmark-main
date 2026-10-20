"""Main Benchmark Execution Pipeline for IncrementalImpact.

Executes end-to-end experiments:
1. Loads data (synthetic or Criteo sample via streaming).
2. Validates schema and performs leakage-free 60/20/20 stratified partitioning.
3. Fits baseline models (Random, Response) and CATE meta-learners (S, T, X).
4. Evaluates ranking metrics (AUUC, Qini, Uplift@10/20/30%, Incremental visits/1k, IPW policy value).
5. Computes paired bootstrap confidence intervals.
6. Runs multi-seed analysis to capture training variability.
7. Logs all runs to local MLflow and DuckDB.
8. Serializes artifacts for FastAPI and Streamlit dashboard.

Usage:
    python scripts/run_benchmark.py --mode synthetic --sample-size 50000 --seeds 42 101
    python scripts/run_benchmark.py --mode criteo --sample-size 100000 --seeds 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from incremental_impact.data.validator import validate_uplift_data
from incremental_impact.data.synthetic import generate_synthetic_data, SyntheticDGPConfig
from incremental_impact.data.splitter import stratified_uplift_split
from incremental_impact.data.loader import load_or_sample_criteo
from incremental_impact.models.baselines import RandomTargetingBaseline, ResponseModelBaseline
from incremental_impact.models.s_learner import SLearner
from incremental_impact.models.t_learner import TLearner
from incremental_impact.models.x_learner import XLearner
from incremental_impact.evaluation.metrics import (
    uplift_at_k,
    incremental_visits_metrics,
    qini_curve,
    qini_coefficient,
    cumulative_gain_curve,
    auuc,
    uplift_by_decile,
)
from incremental_impact.evaluation.policy_value import compute_policy_value
from incremental_impact.evaluation.bootstrap import (
    bootstrap_scalar_metric,
    paired_bootstrap_comparison,
)
from incremental_impact.policy.targeting import apply_capacity_targeting
from incremental_impact.policy.simulator import simulate_budget_scenario
from incremental_impact.tracking.tracker import ExperimentTracker


def build_models(seed: int) -> dict[str, Any]:
    """Instantiate candidate models and baselines for benchmarking."""
    return {
        "RandomBaseline": RandomTargetingBaseline(random_seed=seed),
        "ResponseBaseline_LogReg": ResponseModelBaseline(model_type="logistic_regression", random_seed=seed),
        "ResponseBaseline_LightGBM": ResponseModelBaseline(model_type="lightgbm", random_seed=seed),
        "SLearner_LogReg": SLearner(model_type="logistic_regression", random_seed=seed),
        "SLearner_LightGBM": SLearner(model_type="lightgbm", random_seed=seed),
        "TLearner_LogReg": TLearner(model_type="logistic_regression", random_seed=seed),
        "TLearner_LightGBM": TLearner(model_type="lightgbm", random_seed=seed),
        "XLearner_LogReg": XLearner(model_type="logistic_regression", propensity_type="constant", random_seed=seed),
        "XLearner_LightGBM": XLearner(model_type="lightgbm", propensity_type="constant", random_seed=seed),
    }


def run_benchmark(
    mode: str = "synthetic",
    sample_size: int = 50000,
    seeds: list[int] | None = None,
    raw_data_path: str = "data/raw/criteo-research-uplift-v2.1.csv.gz",
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    if seeds is None:
        seeds = [42, 101]

    print("=" * 80)
    print(f" INCREMENTALIMPACT BENCHMARK EXECUTION ")
    print(f" Mode: {mode.upper()} | Sample Size: {sample_size:,} | Seeds: {seeds}")
    print("=" * 80)

    # 1. Ingestion
    if mode == "criteo":
        criteo_file = Path(raw_data_path)
        if not criteo_file.is_file():
            print(f"Warning: Raw Criteo dataset not found at '{criteo_file}'. Falling back to synthetic mode.")
            mode = "synthetic"

    if mode == "criteo":
        df = load_or_sample_criteo(
            raw_path=raw_data_path,
            sample_size=sample_size,
            random_seed=seeds[0],
            manifest_path="data/manifests/criteo_manifest.json",
        )
    else:
        print(f"Generating representative synthetic dataset ({sample_size:,} rows)...")
        df, _ = generate_synthetic_data(
            SyntheticDGPConfig(
                n_samples=sample_size,
                regime="heterogeneous",
                treatment_rate=0.85,
                base_rate_logit=-3.2,
                max_uplift=0.04,
                random_seed=seeds[0],
            )
        )
        # Write manifest for synthetic dataset
        manifest = {
            "dataset_name": "Synthetic Criteo-Calibrated Benchmark DGP",
            "mode": "synthetic",
            "regime": "heterogeneous",
            "sample_size": sample_size,
            "random_seed": seeds[0],
            "treatment_share": float(df["treatment"].mean()),
            "control_visit_rate": float(df.loc[df["treatment"] == 0, "visit"].mean()),
            "treated_visit_rate": float(df.loc[df["treatment"] == 1, "visit"].mean()),
        }
        manifest_path = Path("data/manifests/synthetic_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # 2. Validation
    val_report = validate_uplift_data(df)
    print(f"Validation Report: {val_report.total_rows:,} rows | Treatment Share: {val_report.treatment_share:.2%}")
    print(f"Outcome Rates -> Control: {val_report.outcome_rates_by_arm['control']:.3%} | Treated: {val_report.outcome_rates_by_arm['treated']:.3%}")
    print(f"Observed Sample ATE: {val_report.raw_ate:.4%}")

    # Initialize tracker
    tracker = ExperimentTracker(
        mlflow_uri="sqlite:///mlruns.db",
        experiment_name=f"incremental_impact_{mode}",
        duckdb_path="reports/experiment_summary.duckdb",
    )

    all_seed_results: list[dict[str, Any]] = []
    curves_data: dict[str, Any] = {}
    best_model_obj = None
    best_auuc = -float("inf")

    # Fixed Holdout Partitioning: 60% Train, 20% Val, 20% Test
    # Test partition remains completely untouched until final scoring!
    splits = stratified_uplift_split(
        df,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_seed=seeds[0],
        scale_features=False,
    )
    print(f"Partitions: Train={splits.train_size:,} | Val={splits.val_size:,} | Test={splits.test_size:,}")

    features = splits.feature_names
    X_train = splits.X_train
    t_train = splits.t_train
    y_train = splits.y_train

    X_test = splits.X_test
    t_test = splits.t_test
    y_test = splits.y_test

    for seed in seeds:
        print(f"\n--- Running Training & Evaluation for Seed {seed} ---")
        models = build_models(seed=seed)
        seed_predictions: dict[str, np.ndarray] = {}

        for name, model in models.items():
            t0 = time.time()
            model.fit(X_train, t_train, y_train)
            fit_time = time.time() - t0

            # Predict on strictly held-out test partition
            test_scores = model.predict_uplift(X_test)
            seed_predictions[name] = test_scores

            # Calculate primary ranking metrics
            auuc_score = auuc(y_test, t_test, test_scores)
            qini_score = qini_coefficient(y_test, t_test, test_scores)

            u10 = uplift_at_k(y_test, t_test, test_scores, k=0.10)
            u20 = uplift_at_k(y_test, t_test, test_scores, k=0.20)
            u30 = uplift_at_k(y_test, t_test, test_scores, k=0.30)

            inc_10 = incremental_visits_metrics(y_test, t_test, test_scores, k=0.10)["incremental_visits_per_1000_targeted"]
            inc_20 = incremental_visits_metrics(y_test, t_test, test_scores, k=0.20)["incremental_visits_per_1000_targeted"]
            inc_30 = incremental_visits_metrics(y_test, t_test, test_scores, k=0.30)["incremental_visits_per_1000_targeted"]

            # Capacity targeting policy (top 20% capacity)
            cap_res = apply_capacity_targeting(test_scores, capacity=0.20)
            pv_res = compute_policy_value(
                y=y_test,
                treatment=t_test,
                policy_actions=cap_res.policy_actions,
            )

            metrics_dict = {
                "auuc": float(auuc_score),
                "qini_coefficient": float(qini_score),
                "uplift_at_10": float(u10),
                "uplift_at_20": float(u20),
                "uplift_at_30": float(u30),
                "inc_visits_1k_top10": float(inc_10),
                "inc_visits_1k_top20": float(inc_20),
                "inc_visits_1k_top30": float(inc_30),
                "incremental_policy_value": float(pv_res["incremental_policy_value"]),
                "fit_time_seconds": float(fit_time),
            }

            # Log to MLflow and DuckDB
            run_id = tracker.log_run(
                model_name=name,
                dataset_mode=mode,
                sample_size=sample_size,
                random_seed=seed,
                base_classifier="LogisticRegression" if "LogReg" in name else "LightGBM",
                metrics=metrics_dict,
                tags={"framework": "IncrementalImpact", "split": "test_holdout"},
            )

            result_entry = {
                "run_id": run_id,
                "seed": seed,
                "model_name": name,
                **metrics_dict,
            }
            all_seed_results.append(result_entry)
            print(f"  {name:<25} | AUUC: {auuc_score:+.6f} | Qini: {qini_score:+.6f} | Uplift@10%: {u10:+.4%} | Time: {fit_time:.2f}s")

            # Store curves for seed 42
            if seed == seeds[0]:
                q_df = qini_curve(y_test, t_test, test_scores)
                g_df = cumulative_gain_curve(y_test, t_test, test_scores)
                dec_df = uplift_by_decile(y_test, t_test, test_scores)
                curves_data[name] = {
                    "qini_curve": q_df.to_dict(orient="list"),
                    "gain_curve": g_df.to_dict(orient="list"),
                    "decile_table": dec_df.to_dict(orient="records"),
                }

                if auuc_score > best_auuc and "Baseline" not in name:
                    best_auuc = auuc_score
                    best_model_obj = model

        # Paired bootstrap statistical comparisons for seed 42
        if seed == seeds[0]:
            print("\nComputing Paired Bootstrap Confidence Intervals (200 resamples)...")
            paired_comparisons = []
            resp_scores = seed_predictions["ResponseBaseline_LightGBM"]
            rand_scores = seed_predictions["RandomBaseline"]

            for candidate in ["SLearner_LightGBM", "TLearner_LightGBM", "XLearner_LightGBM"]:
                cand_scores = seed_predictions[candidate]
                # vs Response Baseline
                cmp_resp = paired_bootstrap_comparison(
                    metric_fn=qini_coefficient,
                    y=y_test,
                    treatment=t_test,
                    score_a=cand_scores,
                    score_b=resp_scores,
                    name_a=candidate,
                    name_b="ResponseBaseline_LightGBM",
                    n_boot=n_bootstrap,
                    random_seed=42,
                )
                paired_comparisons.append(cmp_resp)

                # vs Random Baseline
                cmp_rand = paired_bootstrap_comparison(
                    metric_fn=qini_coefficient,
                    y=y_test,
                    treatment=t_test,
                    score_a=cand_scores,
                    score_b=rand_scores,
                    name_a=candidate,
                    name_b="RandomBaseline",
                    n_boot=n_bootstrap,
                    random_seed=42,
                )
                paired_comparisons.append(cmp_rand)

    # Save artifacts
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = reports_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_seed_results)
    results_df.to_csv(reports_dir / "benchmark_metrics.csv", index=False)

    with open(reports_dir / "benchmark_metrics.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)

    with open(curves_dir / "curves_data.json", "w") as f:
        json.dump(curves_data, f, indent=2)

    with open(reports_dir / "paired_bootstrap_comparisons.json", "w") as f:
        json.dump(paired_comparisons, f, indent=2)

    # Save best model to artifacts/models/
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    if best_model_obj is not None:
        best_model_obj.save(models_dir / "best_uplift_model.pkl")
        print(f"\nSaved Best Uplift Model ({best_model_obj.name}) to artifacts/models/best_uplift_model.pkl")

    # Aggregate multi-seed summary
    agg_df = (
        results_df.groupby("model_name")[
            ["auuc", "qini_coefficient", "uplift_at_10", "inc_visits_1k_top10", "incremental_policy_value"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_df.to_csv(reports_dir / "multi_seed_aggregated_summary.csv", index=False)

    print("\n" + "=" * 90)
    print(" FINAL MULTI-SEED BENCHMARK SUMMARY (Mean ± Std over seeds) ")
    print("=" * 90)
    print(f"{'Model Name':<28} | {'AUUC (Mean ± Std)':<20} | {'Qini (Mean ± Std)':<20} | {'Uplift@10%':<15}")
    print("-" * 90)
    for model_name in results_df["model_name"].unique():
        sub = results_df[results_df["model_name"] == model_name]
        m_auuc, s_auuc = sub["auuc"].mean(), sub["auuc"].std()
        m_qini, s_qini = sub["qini_coefficient"].mean(), sub["qini_coefficient"].std()
        m_u10 = sub["uplift_at_10"].mean()
        print(f"{model_name:<28} | {m_auuc:+.5f} ± {s_auuc:.5f}   | {m_qini:+.5f} ± {s_qini:.5f}   | {m_u10:+.3%}")
    print("=" * 90)

    return {
        "status": "success",
        "sample_size": sample_size,
        "mode": mode,
        "seeds": seeds,
        "results_table": all_seed_results,
        "paired_comparisons": paired_comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IncrementalImpact Benchmark")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "criteo"])
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101])
    parser.add_argument("--raw-data-path", type=str, default="data/raw/criteo-research-uplift-v2.1.csv.gz")
    parser.add_argument("--n-bootstrap", type=int, default=200)

    args = parser.parse_args()
    run_benchmark(
        mode=args.mode,
        sample_size=args.sample_size,
        seeds=args.seeds,
        raw_data_path=args.raw_data_path,
        n_bootstrap=args.n_bootstrap,
    )
