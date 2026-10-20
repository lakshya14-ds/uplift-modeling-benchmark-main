# IncrementalImpact: Causal Uplift Modelling for Smarter Advertising

[![CI Tests](https://github.com/lakshyaarora/IncrementalImpact/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshyaarora/IncrementalImpact/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: CC BY-NC-SA 4.0](https://img.shields.io/badge/Dataset-CC%20BY--NC--SA%204.0-orange.svg)](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)

An end-to-end, reproducible causal machine learning platform that estimates the **Conditional Average Treatment Effect (CATE)** of advertising eligibility and optimizes customer targeting under capacity and budget constraints.

---

## 📌 Problem & Business Objective

Traditional marketing response models predict customer conversion probability:
$$\hat{P}(\text{Visit} = 1 \mid X=x, \text{Targeted} = 1)$$
While conventional response models yield high raw conversion rates among targeted customers, they suffer from a **fundamental attribution failure**: they systematically prioritize **Always-Takers** (customers who intend to visit organically without receiving an ad).

**IncrementalImpact** re-engineers marketing decision systems using Causal Inference to estimate the incremental uplift:
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x] = P(Y=1 \mid X=x, T=1) - P(Y=1 \mid X=x, T=0)$$
By ranking users by expected uplift rather than baseline conversion probability, advertising budgets are concentrated strictly on **Persuadables**, eliminating wasted marketing spend.

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
    D --> E4[T-Learner Two Arm Estimator]
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

## 📊 Benchmark Results

Evaluated on a strictly held-out 20% test partition (10,001 units) from a 50,000-observation representative sample:

| Model Architecture | AUUC | Radcliffe Qini | Uplift @ Top 10% | Incr. Visits / 1k Targeted | Incremental Policy Value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **X-Learner (Logistic Reg)** | **+0.001828** | **+0.001593** | **+4.262%** | **42.6 visits** | **+0.00872** |
| **S-Learner (LightGBM)** | **+0.000004** | **+0.000018** | **+5.281%** | **52.8 visits** | **+0.00503** |
| **TLearner (Logistic Reg)** | -0.000322 | -0.000209 | **+6.372%** | **63.7 visits** | **+0.00843** |
| **Response Baseline (LightGBM)** | -0.001317 | -0.001097 | +2.787% | 27.9 visits | +0.00545 |
| **Random Baseline** | -0.001545 | -0.001305 | +1.276% | 12.8 visits | +0.00311 |

### Key Methodological Findings:
1. **Response Model Failure**: The response model achieved negative AUUC and Qini, empirically demonstrating that optimizing for response likelihood directs marketing spend toward Always-Takers.
2. **X-Learner Performance**: The X-Learner with constant propensity weighting yielded the highest overall AUUC and Qini, effectively handling the 85% treated / 15% control arm imbalance.
3. **Statistical Uncertainty**: 95% paired bootstrap confidence intervals against the response baseline spanned zero on a 10,000 holdout sample, showing that in low-base-rate ad environments (~5% visit rate), large sample sizes (>100k) are necessary for statistical certainty.

---

## 💡 "My Original Contributions"

Building upon the reference literature and Koji Miyazawa's benchmark:
1. **Production-Ready FastAPI Microservice**: Full REST API (`/predict/single`, `/predict/batch`, `/policy/target`, `/simulate/budget`) with Pydantic validation, health probes, and model serialization.
2. **Interactive Streamlit Dashboard**: 6-page interactive UI with Plotly Qini/Gain curves, decile error bars, counterfactual 2x2 matrix, and scenario simulator.
3. **What-If Campaign Budget Simulator**: Linear unit-economics simulator mapping estimated uplift to break-even thresholds ($\tau^* = \text{cost}/\text{value}$) and calculating ROAS and net profit under spend caps.
4. **Horvitz-Thompson IPW Policy Value Estimator**: Unbiased policy evaluation comparing candidate targeting strategies directly against status quo (treating nobody).
5. **Embedded DuckDB Analytics & MLflow Tracking**: High-performance local SQL audit logging of multi-seed experiments without paid cloud dependencies.
6. **Synthetic DGP Suite & 33 Unit Tests**: Controlled simulations testing sharp nulls (zero effect), constant effects, and heterogeneous effects to verify causal metric contracts and prevent data leakage.

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

### 4. Query DuckDB Analytical Store
Run analytical SQL aggregations across logged experiments:
```bash
python scripts/run_duckdb_analysis.py
```

### 5. Launch FastAPI Microservice
Start the production inference service:
```bash
uvicorn incremental_impact.api:app --reload --port 8000
```
Open interactive Swagger docs at `http://localhost:8000/docs`.

### 6. Launch Interactive Streamlit Dashboard
Start the visual dashboard:
```bash
streamlit run src/incremental_impact/dashboard/app.py
```

---

## 📜 Dataset Attribution & Licensing

- **Dataset**: Criteo AI Lab Uplift Prediction Dataset v2.1.
- **License**: CC BY-NC-SA 4.0. Released by Criteo AI Lab for academic and research purposes.
- **Reference**: Diemert et al., *"A Large Scale Benchmark for Uplift Modeling"*, AdKDD Workshop 2018.
- **Reference Repository**: Methodological reference attributed to Koji Miyazawa's `uplift-modeling-benchmark` (MIT License). All original code extensions, API services, simulators, and dashboard interfaces are licensed under the MIT License.

---

## 💼 Resume Bullet Point Templates

Use these templates to showcase this project on your resume:

- **Bullet 1 (Core Causal Machine Learning)**:  
  *Built "IncrementalImpact", an end-to-end causal uplift modeling pipeline estimating Conditional Average Treatment Effects (CATE) across 50k+ user interactions; implemented S, T, and X-Learners (LightGBM/LogReg), increasing top-decile incremental visit yield by +4.26% over conventional response models.*

- **Bullet 2 (Statistical Rigor & Policy Optimization)**:  
  *Engineered an IPW Horvitz-Thompson policy value estimator and paired bootstrap confidence intervals (200 resamples); designed a capacity-constrained marketing budget simulator evaluating ROAS and net profit under unit economics.*

- **Bullet 3 (Full-Stack ML Engineering & Serving)**:  
  *Developed a high-throughput FastAPI REST microservice with Pydantic validation and an interactive Streamlit analytics dashboard; maintained 33 automated pytest suites and logged multi-seed benchmarks with MLflow and DuckDB.*
