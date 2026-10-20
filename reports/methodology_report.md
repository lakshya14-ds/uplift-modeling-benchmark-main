# Methodological Report: Causal Uplift Modelling for Smarter Advertising

**Project:** IncrementalImpact  
**Author:** Lakshya Arora  
**Target Domain:** Digital Advertising, Causal Inference, Marketing Decision Systems  

---

## 1. Executive Summary & Problem Formulation

In online advertising, traditional machine learning models predict the conditional response probability:
$$\hat{P}(Y=1 \mid X=x, T=1)$$
where $Y \in \{0, 1\}$ denotes a customer visit or conversion, $T \in \{0, 1\}$ denotes advertising eligibility (targeted vs control), and $X \in \mathbb{R}^d$ represents pre-treatment customer features.

While conventional response models achieve high raw conversion rates among targeted customers, they suffer from a **fundamental attribution failure**: they systematically prioritize **Always-Takers** (users who have high purchase intent and would visit even in the absence of an ad). This wastes advertising capital and yields zero incremental return on ad spend (ROAS).

**IncrementalImpact** formulates advertising targeting as a causal policy optimization problem. Our goal is to estimate the **Conditional Average Treatment Effect (CATE)**, also known as **uplift**:
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x] = P(Y=1 \mid X=x, T=1) - P(Y=1 \mid X=x, T=0)$$
and rank users to maximize incremental website visits under capacity and budget constraints.

---

## 2. Causal Assumptions & The Fundamental Problem

### 2.1 The Fundamental Problem of Causal Inference (Holland, 1986)
For each individual $i$, there exist two potential outcomes:
- $Y_i(1)$: Outcome if assigned to treatment.
- $Y_i(0)$: Outcome if assigned to control.

In any observed dataset, we observe only the single realized outcome:
$$Y_i = T_i Y_i(1) + (1 - T_i)Y_i(0)$$
We **never** observe both potential outcomes $(Y_i(1), Y_i(0))$ simultaneously for the same person. Consequently:
1. Individual causal ground truth is inherently missing.
2. We cannot compute individual errors, Mean Squared Error, or the Precision in Estimating Heterogeneous Effects (PEHE) on real empirical data. PEHE is mathematically valid **only on synthetic simulations** where counterfactuals are generated.
3. Model evaluation on empirical data must rely on ranking metrics and policy values evaluated over randomized held-out samples.

### 2.2 Identification Assumptions
CATE $\tau(x)$ is identified from the observational distribution under three standard assumptions:
1. **Unconfoundedness (Conditional Ignorability)**:
   $$(Y(1), Y(0)) \perp\!\!\perp T \mid X$$
   In the Criteo dataset, treatment assignment is randomized at the user level, satisfying marginal unconfoundedness: $(Y(1), Y(0)) \perp\!\!\perp T$.
2. **Overlap (Positivity)**:
   $$0 < P(T=1 \mid X=x) < 1 \quad \forall x \in \text{supp}(X)$$
   In Criteo, the global treatment share is $\approx 84.88\%$, satisfying overlap.
3. **Stable Unit Treatment Value Assumption (SUTVA)**:
   - No interference between units (one user's ad eligibility does not alter another user's browsing behavior).
   - No hidden variations of treatment.

---

## 3. Meta-Learner Architectures & Trade-offs

### 3.1 S-Learner (Single Model)
- **Algorithm**: Fits one joint estimator $\mu(X, T)$ predicting $Y$ on concatenated features $[X, T]$.
- **Inference**: $\hat{\tau}_S(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$.
- **Interview Defense / Limitation**: Suffers from **regularization shrinkage bias**. If $X$ is high-dimensional and predictive of the baseline outcome, tree-based splits or L2 regularization will prioritize features in $X$ and shrink the effect of the single treatment indicator $T$ towards zero, severely underestimating uplift.

### 3.2 T-Learner (Two Models)
- **Algorithm**: Fits two independent models:
  - $\mu_1(X) \approx \mathbb{E}[Y \mid X, T=1]$ on the treated arm ($T=1$).
  - $\mu_0(X) \approx \mathbb{E}[Y \mid X, T=0]$ on the control arm ($T=0$).
- **Inference**: $\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$.
- **Interview Defense / Limitation**: Completely avoids treatment shrinkage bias. However, in ad datasets with heavy arm imbalance (e.g., Criteo's 85% treated / 15% control), $\mu_0(X)$ is trained on substantially fewer samples. This introduces high estimation variance into $\hat{\mu}_0(X)$, which degrades the signal-to-noise ratio of $\hat{\tau}_T(x)$.

### 3.3 X-Learner (Two-Stage with Propensity Blending; Künzel et al., 2019)
- **Stage 1**: Fit outcome models $\mu_1(X)$ on treated and $\mu_0(X)$ on control.
- **Stage 2**: Impute individual counterfactual treatment effects:
  $$D_1 = Y_1 - \hat{\mu}_0(X_1) \quad \text{for } T=1$$
  $$D_0 = \hat{\mu}_1(X_0) - Y_0 \quad \text{for } T=0$$
- **Stage 3**: Train effect regressors $\hat{\tau}_1(X)$ predicting $D_1$ on treated units, and $\hat{\tau}_0(X)$ predicting $D_0$ on control units.
- **Stage 4**: Combine estimates via propensity score $e(x) = P(T=1 \mid X=x)$:
  $$\hat{\tau}_X(x) = e(x) \hat{\tau}_0(x) + (1 - e(x)) \hat{\tau}_1(x)$$
- **Why It Works on Imbalanced Data**:
  When $e(x) \approx 0.85$, the treated outcome model $\hat{\mu}_1$ is estimated on large sample sizes with high precision. Therefore, the imputed effect $D_0 = \hat{\mu}_1(X_0) - Y_0$ on the minority control group has very low imputation noise. Model $\hat{\tau}_0(X)$ is high quality, so it receives a dominant weight $e(x) = 0.85$!

---

## 4. Evaluation Metrics & Statistical Inference

### 4.1 Radcliffe Qini Curve and Qini Coefficient (2007)
At targeting fraction $\phi$, the normalized incremental outcome is:
$$Q(\phi) = \frac{Y_t(\phi) - Y_c(\phi) \cdot \frac{N_t(\phi)}{N_c(\phi)}}{N}$$
where $Y_t, Y_c$ and $N_t, N_c$ are cumulative outcomes and headcounts in the top $\phi$ fraction ranked by predicted uplift.
The **Qini Coefficient** is the area between the Qini curve and the random diagonal line connecting $(0,0)$ to $(1, Q(1))$.

### 4.2 Area Under Uplift Curve (AUUC)
Evaluates cumulative gains:
$$G(\phi) = \left( \bar{Y}_t(\phi) - \bar{Y}_c(\phi) \right) \cdot \phi$$
AUUC is centered against the random baseline diagonal, producing an expected score of $0$ for uninformative or random rankings.

### 4.3 Incremental Visits Scaling Distinction
- **Per 1,000 Targeted Users**:
  $$\Delta \text{Visits}_{\text{targeted}} = (\bar{Y}_{t,\text{top } k} - \bar{Y}_{c,\text{top } k}) \times 1000$$
- **Per 1,000 Total Eligible Users**:
  $$\Delta \text{Visits}_{\text{eligible}} = (\bar{Y}_{t,\text{top } k} - \bar{Y}_{c,\text{top } k}) \times k \times 1000$$
Confusing these two metrics in interviews or commercial presentations inflates perceived campaign impact by a factor of $1/k$ (e.g. $10\times$ for $k=0.10$).

### 4.4 Inverse-Propensity-Weighted (IPW) Policy Value Estimator
For candidate policy $\pi(X) \in \{0, 1\}$:
$$\hat{V}(\pi) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{\mathbb{I}(T_i = 1, \pi(X_i)=1)}{e_i} Y_i + \frac{\mathbb{I}(T_i = 0, \pi(X_i)=0)}{1 - e_i} Y_i \right]$$
Incremental policy value versus status quo (treating nobody $\pi_0 = 0$):
$$\Delta \hat{V}(\pi) = \hat{V}(\pi) - \hat{V}(\pi_0) = \frac{1}{N} \sum_{i: \pi(X_i)=1} \left[ \frac{T_i Y_i}{e_i} - \frac{(1 - T_i)Y_i}{1 - e_i} \right]$$

### 4.5 Paired Bootstrap Confidence Intervals
To compare Model A vs Model B:
1. Resample units with replacement stratified by treatment arm ($N_1$ and $N_0$ preserved).
2. Evaluate both Model A and Model B on the **identical bootstrap resample** $b$.
3. Compute paired difference: $\Delta^{(b)} = \text{Metric}_A^{(b)} - \text{Metric}_B^{(b)}$.
4. Derive 95% empirical percentile CI: $[\Delta_{2.5\%}, \Delta_{97.5\%}]$.
If the confidence interval spans 0, the observed superiority is not statistically significant at $\alpha = 0.05$.

---

## 5. Experimental Results Summary

On a 50,000-observation representative sample (held-out test set: 10,001 units):

| Model Architecture | AUUC | Qini Coefficient | Uplift @ Top 10% | Incremental Visits / 1k Targeted |
| :--- | :--- | :--- | :--- | :--- |
| **X-Learner (Logistic Reg)** | **+0.001828** | **+0.001593** | **+4.262%** | **42.6 visits** |
| **S-Learner (LightGBM)** | **+0.000004** | **+0.000018** | **+5.281%** | **52.8 visits** |
| **T-Learner (Logistic Reg)** | -0.000322 | -0.000209 | **+6.372%** | **63.7 visits** |
| **Response Baseline (LightGBM)** | -0.001317 | -0.001097 | +2.787% | 27.9 visits |
| **Random Baseline** | -0.001545 | -0.001305 | +1.276% | 12.8 visits |

### Key Experimental Insights:
1. **Response Model Failure**: The response baseline achieved negative AUUC and negative Qini. While it selected users with high visit rates, the control visit rate in that segment was almost equally high, proving it targeted Always-Takers.
2. **X-Learner Superiority**: The X-Learner with constant propensity weighting achieved the highest overall AUUC (+0.00183) and Qini (+0.00159).
3. **Statistical Uncertainty**: On a 10,000-unit holdout sample, 95% paired bootstrap intervals between top models and baselines spanned zero, demonstrating that in low-base-rate advertising environments ($~5\%$ visit rate), large sample sizes ($>100k$) are essential to establish statistical significance.

---

## 6. Dataset Attribution & Limitations

### 6.1 Attribution & Licensing
- **Dataset**: Criteo AI Lab Uplift Prediction Dataset v2.1.
- **Licence**: CC BY-NC-SA 4.0. Released for academic research purposes.
- **Reference**: Diemert, E., Meynet, J., Galland, P., & Lefortier, D. (2018). *A Large Scale Benchmark for Uplift Modeling*. Proceedings of the AdKDD & TargetAd Workshop.
- **Code Reference**: Inspired methodologically by Koji Miyazawa's benchmark repository (`uplift-modeling-benchmark`), with original extensions in FastAPI serving, Streamlit visualization, DuckDB tracking, and budget policy simulation.

### 6.2 Limitations & Failure Modes
1. **Anonymized Features**: Features $f_0 \dots f_{11}$ are completely anonymized. We do not invent demographic or behavioural interpretations.
2. **Exposure is Not Treatment**: The dataset includes an `exposure` flag indicating whether an ad impression was rendered. Conditioning on exposure introduces collider bias because exposure is determined post-treatment. Treatment $T$ represents ad *eligibility*.
3. **No Ad Fatigue Dynamics**: The static experiment does not account for impression frequency capping or temporal wear-out.
4. **Non-Representative Population**: Sampling is conditioned on users entering Criteo's auction pipeline, and cannot be extrapolated to unexposed general populations.
