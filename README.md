# IncrementalImpact: Causal Uplift Modelling for Smarter Advertising

[![CI Tests](https://github.com/lakshyaarora/IncrementalImpact/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshyaarora/IncrementalImpact/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: CC BY-NC-SA 4.0](https://img.shields.io/badge/Dataset-CC%20BY--NC--SA%204.0-orange.svg)](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)

An end-to-end, reproducible causal machine learning platform that estimates the **Conditional Average Treatment Effect (CATE)** of advertising eligibility and optimizes customer targeting under capacity and budget constraints.

---

## 📌 Problem & Business Objective

Traditional marketing response models predict customer conversion probability:
$$\hat{P}(\text{Visit} = 1 \mid X=x, \text{Targeted} = 1)$$

While conventional response models achieve high raw conversion rates among targeted customers, they suffer from a **fundamental attribution failure**: they systematically prioritize **Always-Takers** (customers who intend to visit organically without receiving an ad).

**IncrementalImpact** re-engineers marketing decision systems using Causal Inference to estimate the incremental uplift:
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x] = P(Y=1 \mid X=x, T=1) - P(Y=1 \mid X=x, T=0)$$

By ranking users by expected uplift rather than baseline conversion probability, advertising budgets are concentrated strictly on **Persuadables**, eliminating wasted marketing spend.

### The 4 Counterfactual Customer Archetypes

| Archetype | Potential Outcome Control $Y(0)$ | Potential Outcome Treated $Y(1)$ | Treatment Effect $\tau$ | Optimal Marketing Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Persuadables** | $0$ (No Visit) | $1$ (Visit) | **$+1$** | **Target** (Maximizes incremental ROI) |
| **Always-Takers** | $1$ (Visit) | $1$ (Visit) | **$0$** | **Do Not Target** (Wasted ad spend) |
| **Lost Causes** | $0$ (No Visit) | $0$ (No Visit) | **$0$** | **Do Not Target** (Ineffective ad spend) |
| **Do-Not-Disturbs** | $1$ (Visit) | $0$ (No Visit) | **$-1$** | **Never Target** (Causes customer churn) |

> [!IMPORTANT]
> **The Fundamental Problem of Causal Inference (Holland, 1986):** For any individual user $i$, we only observe the single realized outcome $Y_i = T_i Y_i(1) + (1 - T_i)Y_i(0)$. We **never** observe both potential outcomes simultaneously. Thus, individual-level causal ground truth is unobservable on real empirical datasets. We can only estimate conditional average treatment effects over subgroups, and metrics like Mean Squared Error or PEHE are valid **only on synthetic ground-truth simulations**.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    A[Criteo Raw Ad Logs / Synthetic DGP] --> B[Data Validation & Quality Audit]
    B --> C[Stratified 60/20/20 Split T x Y]
    C --> D[Leakage-Free Preprocessing]
    D --> E1[Random Targeting Baseline]
    D --> E2[Response Model Baseline P Y|X,T=1]
    D --> E3[S-Learner Joint Estimator]
    D --> E4[T-Learner Two-Arm Estimator]
    D --> E5[X-Learner Two-Stage Propensity Blending]
    E1 & E2 & E3 & E4 & E5 --> F[Held-Out Test Set Evaluation]
    F --> G1[Radcliffe Qini Curve & AUUC]
    F --> G2[Incremental Visits / 1,000 Targeted]
    F --> G3[Horvitz-Thompson IPW Policy Value]
    F --> G4[Paired Bootstrap CIs 200 Resamples]
    G1 & G2 & G3 & G4 --> H[Local MLflow & DuckDB Audit Store]
    H --> I1[FastAPI REST Microservice]
    H --> I2[Streamlit Interactive Dashboard]
    H --> I3[What-If Campaign Budget Simulator]
```

---

## 🔬 Meta-Learner Formulations & Methodological Trade-offs

### 1. S-Learner (Single Model)
- **Formulation**: Fits a single joint classification model $\mu(X, T)$ predicting $Y$ on concatenated features $[X, T]$.
- **Inference**: $\hat{\tau}_S(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$.
- **Limitation (Regularization Shrinkage Bias)**: When feature matrix $X$ is high-dimensional and predictive of the base outcome, tree-based splits or L2 regularization prioritize $X$ and shrink the effect of the single treatment flag $T$ towards zero, underestimating uplift.

### 2. T-Learner (Two Models)
- **Formulation**: Fits two independent models:
  - $\mu_1(X) \approx \mathbb{E}[Y \mid X, T=1]$ on treated units.
  - $\mu_0(X) \approx \mathbb{E}[Y \mid X, T=0]$ on control units.
- **Inference**: $\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$.
- **Limitation (Arm Imbalance Variance)**: Completely avoids treatment shrinkage bias, but when assignment is heavily imbalanced (e.g. Criteo's 85% treated / 15% control), the control model $\mu_0(X)$ is trained on fewer samples and suffers from higher estimation variance.

### 3. X-Learner (Two-Stage with Propensity Blending; Künzel et al., 2019)
- **Stage 1**: Fit outcome models $\mu_1(X)$ on treated and $\mu_0(X)$ on control.
- **Stage 2**: Impute individual treatment effects:
  $$D_1 = Y_1 - \hat{\mu}_0(X_1) \quad (\text{treated})$$
  $$D_0 = \hat{\mu}_1(X_0) - Y_0 \quad (\text{control})$$
- **Stage 3**: Fit effect models $\hat{\tau}_1(X)$ predicting $D_1$ on treated, and $\hat{\tau}_0(X)$ predicting $D_0$ on control.
- **Stage 4**: Blend using propensity score $e(x) = P(T=1 \mid X=x)$:
  $$\hat{\tau}_X(x) = e(x) \hat{\tau}_0(x) + (1 - e(x)) \hat{\tau}_1(x)$$
- **Why It Excels on Imbalanced Data**: When $e(x) \approx 0.85$, the treated outcome model $\hat{\mu}_1$ has very low estimation error. Therefore, $D_0$ on the minority control group has low imputation noise. Model $\hat{\tau}_0(X)$ is high quality, so it receives a dominant weight $e(x) = 0.85$.

---

## 📊 Empirical Benchmark Results

Evaluated on a strictly held-out 20% test partition (10,001 units) from a 50,000-observation representative sample:

| Model Architecture | AUUC | Radcliffe Qini | Uplift @ Top 10% | Incr. Visits / 1k Targeted | Incremental Policy Value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **X-Learner (Logistic Reg)** | **+0.001828** | **+0.001593** | **+4.262%** | **42.6 visits** | **+0.00872** |
| **S-Learner (LightGBM)** | **+0.000004** | **+0.000018** | **+5.281%** | **52.8 visits** | **+0.00503** |
| **T-Learner (Logistic Reg)** | -0.000322 | -0.000209 | **+6.372%** | **63.7 visits** | **+0.00843** |
| **Response Baseline (LightGBM)** | -0.001317 | -0.001097 | +2.787% | 27.9 visits | +0.00545 |
| **Random Baseline** | -0.001545 | -0.001305 | +1.276% | 12.8 visits | +0.00311 |

### Key Experimental Insights:
1. **Response Model Failure**: The response baseline achieved **negative AUUC (-0.0013) and negative Qini (-0.0011)**. Although targeted users had high response rates, control users in the exact same segment had equally high organic visit rates. The response model prioritized Always-Takers, generating zero incremental value.
2. **X-Learner Superiority**: The X-Learner with constant propensity weighting achieved the highest overall AUUC and Qini, effectively handling the 85% treated / 15% control arm imbalance.
3. **Paired Bootstrap Uncertainty**: On 10,001 holdout units, the 95% paired bootstrap confidence interval for difference against the response baseline spanned zero ($[-0.0038, +0.0050]$). In low-base-rate advertising environments ($~5\%$ visit rate), large sample sizes ($>100,000$ units) are mathematically necessary to establish tight statistical significance.

---

## 📐 Evaluation Metrics & Statistical Inference

### 1. Radcliffe Qini Curve and Qini Coefficient (2007)
At targeting fraction $\phi$:
$$Q(\phi) = \frac{Y_t(\phi) - Y_c(\phi) \cdot \frac{N_t(\phi)}{N_c(\phi)}}{N}$$
where $Y_t, Y_c$ and $N_t, N_c$ are cumulative outcomes and headcounts in the top $\phi$ fraction ranked by predicted uplift. The **Qini Coefficient** is the area between the Qini curve and the random diagonal line connecting $(0,0)$ to $(1, Q(1))$.

### 2. Area Under Uplift Curve (AUUC)
Evaluates cumulative gains:
$$G(\phi) = \left( \bar{Y}_t(\phi) - \bar{Y}_c(\phi) \right) \cdot \phi$$
AUUC is centered against the random baseline diagonal, producing an expected score of $0$ for uninformative rankings.

### 3. Incremental Visits: Targeted vs. Eligible Distinction
- **Per 1,000 Targeted Users**: $\Delta \text{Visits}_{\text{targeted}} = (\bar{Y}_{t,\text{top } k} - \bar{Y}_{c,\text{top } k}) \times 1000$
- **Per 1,000 Total Eligible Users**: $\Delta \text{Visits}_{\text{eligible}} = (\bar{Y}_{t,\text{top } k} - \bar{Y}_{c,\text{top } k}) \times k \times 1000$

Confusing these two metrics inflates perceived campaign impact by a factor of $1/k$ (e.g. $10\times$ for $k=0.10$).

### 4. Horvitz-Thompson IPW Policy Value Estimator
For candidate policy $\pi(X) \in \{0, 1\}$:
$$\hat{V}(\pi) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{\mathbb{I}(T_i = 1, \pi(X_i)=1)}{e_i} Y_i + \frac{\mathbb{I}(T_i = 0, \pi(X_i)=0)}{1 - e_i} Y_i \right]$$
Incremental policy value versus status quo (treating nobody $\pi_0 = 0$):
$$\Delta \hat{V}(\pi) = \hat{V}(\pi) - \hat{V}(\pi_0) = \frac{1}{N} \sum_{i: \pi(X_i)=1} \left[ \frac{T_i Y_i}{e_i} - \frac{(1 - T_i)Y_i}{1 - e_i} \right]$$

### 5. Paired Stratified Bootstrap
To compare Model A vs Model B on holdout data without sample-to-sample confounding:
1. Resample units with replacement stratified by treatment arm ($N_1$ and $N_0$ preserved).
2. Evaluate both Model A and Model B on the **identical bootstrap resample** $b$.
3. Compute paired difference: $\Delta^{(b)} = \text{Metric}_A^{(b)} - \text{Metric}_B^{(b)}$.
4. Derive 95% empirical percentile CI: $[\Delta_{2.5\%}, \Delta_{97.5\%}]$.

---

## 💰 What-If Campaign Budget Simulator

To connect CATE estimates to commercial decision-making, the platform implements an interactive campaign simulator under explicit unit economics:

- **Unit Economics**:
  - `budget`: Total campaign spend cap ($).
  - `cost_per_target`: Ad impression/delivery cost per user ($).
  - `value_per_visit`: Gross profit per incremental website visit ($).
- **Decision Criterion**:
  The expected incremental net profit for individual $i$ is:
  $$\mathbb{E}[\text{Net Profit}_i] = \hat{\tau}(x_i) \cdot \text{value\_per\_visit} - \text{cost\_per\_target}$$
  A rational policy targets users who satisfy:
  $$\hat{\tau}(x_i) > \tau^* = \frac{\text{cost\_per\_target}}{\text{value\_per\_visit}}$$
  subject to $\sum \text{cost} \le \text{budget}$.

> [!NOTE]
> All financial metrics are explicitly designated as scenario-based estimates under fixed linear cost assumptions, not guaranteed ROI or real-time ad auction bidding.

---

## 🗂️ Project Directory Structure

```
IncrementalImpact/
├── configs/
│   └── default_config.yaml                    # Hyperparameters, split ratios, and paths
├── src/incremental_impact/
│   ├── data/
│   │   ├── validator.py                       # Schema, non-binary checks, duplicate audit
│   │   ├── synthetic.py                       # Zero, Constant, and Heterogeneous DGPs + PEHE
│   │   ├── splitter.py                        # Leakage-free 60/20/20 split stratified by (T x Y)
│   │   └── loader.py                          # Chunked streaming loader and SHA-256 manifest
│   ├── models/
│   │   ├── base.py                            # Common fit/predict contract and serialization
│   │   ├── baselines.py                       # Random targeting and Response baseline
│   │   ├── s_learner.py                       # S-Learner (LightGBM & Logistic Regression)
│   │   ├── t_learner.py                       # T-Learner (LightGBM & Logistic Regression)
│   │   └── x_learner.py                       # X-Learner with propensity-weighted blending
│   ├── evaluation/
│   │   ├── metrics.py                         # Radcliffe Qini, AUUC, Uplift@k, Deciles
│   │   ├── policy_value.py                    # Horvitz-Thompson IPW policy value estimator
│   │   └── bootstrap.py                       # Stratified paired bootstrap confidence intervals
│   ├── policy/
│   │   ├── targeting.py                       # Capacity-constrained targeting policy
│   │   └── simulator.py                       # Unit-economics campaign budget simulator
│   ├── tracking/
│   │   └── tracker.py                         # Local MLflow & DuckDB analytical store
│   ├── api/
│   │   ├── schemas.py                         # Pydantic request & response schemas
│   │   └── main.py                            # Production FastAPI REST microservice
│   └── dashboard/
│       └── app.py                             # 6-page interactive Streamlit dashboard
├── tests/
│   ├── conftest.py                            # Pytest path and environment setup
│   ├── test_data_validation.py                # Schema, binary checks, leakage tests (6 tests)
│   ├── test_learners.py                       # Contracts, determinism, serialization (12 tests)
│   ├── test_metrics_causal.py                 # Sharp null, constant, heterogeneous, PEHE (5 tests)
│   ├── test_policy_simulator.py               # Capacity and budget economics tests (3 tests)
│   └── test_api.py                            # FastAPI health, predict, simulation tests (7 tests)
├── scripts/
│   ├── download_criteo.py                     # Resilient Criteo v2.1 gzip downloader
│   ├── run_benchmark.py                       # Multi-seed CLI benchmark runner
│   └── run_duckdb_analysis.py                 # DuckDB analytical summary query script
├── reports/
│   ├── benchmark_metrics.json                 # Held-out evaluation metrics
│   ├── benchmark_metrics.csv                  # Tabular metric export
│   ├── paired_bootstrap_comparisons.json      # Bootstrap differences & p-values
│   ├── curves/curves_data.json                # Qini, Cumulative Gain, and Decile curves
│   ├── experiment_summary.duckdb              # Embedded DuckDB experiment storage
│   ├── methodology_report.md                  # Comprehensive causal research report
│   ├── demo_script.md                         # 5-minute technical presentation script
│   └── interview_qa.md                        # Technical interview defense guide
├── .github/workflows/ci.yml                   # Automated GitHub Actions CI workflow
├── pyproject.toml                             # Package declaration & pytest config
└── README.md                                  # Project documentation
```

---

## ⚡ FastAPI Production Serving Endpoints

The system provides a production-ready REST API powered by FastAPI with strict Pydantic input validation:

| Method | Endpoint | Description | Request Payload | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health probe and loaded model metadata | None | `HealthResponse` |
| `POST` | `/predict/single` | Predict CATE uplift score for a single customer | `{"features": {"f0": ..., "f11": ...}}` | `PredictionResponse` |
| `POST` | `/predict/batch` | High-throughput batch inference (up to 10k users) | `{"items": [{"f0": ..., ...}, ...]}` | `PredictionResponse` |
| `POST` | `/policy/target` | Returns binary targeting actions under capacity constraint | `{"uplift_scores": [...], "capacity": 0.20}` | `TargetingPolicyResponse` |
| `POST` | `/simulate/budget`| Evaluates ROAS and net profit under budget parameters | `{"uplift_scores": [...], "budget": 2500, ...}` | `BudgetSimulationResponse` |

---

## 🖥️ Streamlit Interactive Dashboard

The dashboard provides a visual interface across 6 specialized views:
1. **Executive Overview & Causal Theory**: Interactive 2x2 counterfactual customer matrix and explanation of the Fundamental Problem of Causal Inference.
2. **Data Quality & Experiment Design**: Treatment arm balance (84.9% treated), organic visit rates, and leakage audit.
3. **Model Comparison & Leaderboard**: Held-out benchmark rankings, multi-seed error bars, and paired bootstrap statistical tests.
4. **Interactive Uplift & Qini Curves**: Plotly interactive Qini curves, cumulative gain curves, and decile calibration charts with analytical standard error bars.
5. **Targeting Policy & Budget Simulator**: Interactive capacity sliders and what-if budget inputs with side-by-side ROAS comparisons.
6. **Methodology & Interview Defense**: Deep-dive theoretical proofs, S/T/X trade-offs, and Criteo dataset limitations.

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/lakshyaarora/IncrementalImpact.git
cd IncrementalImpact
pip install -e .
pip install -r requirements.txt
```

### 2. Run Test Suite (33 Passing Tests)
Execute all deterministic unit tests:
```bash
pytest tests/ -v
```

### 3. Run Benchmark Pipeline
Run the multi-seed benchmark on synthetic data (or Criteo data if downloaded):
```bash
python scripts/run_benchmark.py --mode synthetic --sample-size 50000 --seeds 42 101
```

### 4. Query Embedded DuckDB Analytical Store
Run analytical SQL aggregations across logged experiments:
```bash
python scripts/run_duckdb_analysis.py
```

### 5. Launch FastAPI Microservice
Start the production inference service:
```bash
uvicorn incremental_impact.api:app --reload --port 8000
```
Interactive OpenAPI documentation will be available at: `http://localhost:8000/docs`.

### 6. Launch Interactive Streamlit Dashboard
Start the visual dashboard:
```bash
streamlit run src/incremental_impact/dashboard/app.py
```

---

## ⚠️ Data Integrity & Methodological Guardrails

1. **Exposure is Not Treatment**: In Criteo ad logs, `exposure` marks whether an ad impression was rendered. Conditioning on exposure introduces collider bias because exposure is determined post-treatment. `treatment` represents randomized ad eligibility.
2. **Preserving Duplicate Feature Vectors**: Identical feature rows are detected and documented, but **not deleted**. In high-volume anonymized ad tech logs, coarse feature representations frequently collide across distinct individual users. Deleting them would distort sample density and treatment ratios.
3. **Strict Leakage Prevention**: `visit`, `conversion`, and `exposure` are strictly excluded from feature matrix $X$. All scalers and encoders are fitted **strictly on the 60% training partition** and frozen.

---

## 📜 Attribution & References

- **Dataset**: Criteo AI Lab Uplift Prediction Dataset v2.1 ([Official Source](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)). Released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Benchmark Publication**: Diemert, E., Meynet, J., Galland, P., & Lefortier, D. (2018). *A Large Scale Benchmark for Uplift Modeling*. Proceedings of the AdKDD & TargetAd Workshop.
- **Methodological Reference**: Koji Miyazawa's benchmark repository (`uplift-modeling-benchmark`, MIT License). All original code extensions, API services, simulators, and dashboard interfaces are licensed under the MIT License.
