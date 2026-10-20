# Technical Interview Defense Guide: Causal Uplift Modelling

This document contains in-depth, rigorous answers to tough questions you will encounter in Data Science, Machine Learning, and Causal Inference interviews.

---

### Q1: Why not just use a high-capacity binary classifier (e.g. LightGBM) to predict clicks or visits?
**Answer:**
A standard binary classifier models the conditional response distribution:
$$\hat{P}(Y=1 \mid X=x, T=1)$$
This predicts *correlation* with the outcome, not *causation*. In advertising, high-intent users (e.g., users who frequently browse an advertiser's website or search for brand terms) have high visit probabilities regardless of whether they receive an ad ($Y(0) = 1$). 
Targeting them produces high observed conversion rates, but generates zero **incremental** conversion. In fact, targeting can even irritate users (Do-Not-Disturbs / Sleeping Dogs), yielding a negative treatment effect. 
Causal uplift modeling explicitly targets the Conditional Average Treatment Effect (CATE):
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]$$
This guarantees that marketing spend is directed toward users whose behavior is positively modified by the ad (Persuadables).

---

### Q2: What is the Fundamental Problem of Causal Inference, and why can't we compute MSE or PEHE on real data?
**Answer:**
According to Holland (1986), for every individual unit $i$, there exist two potential outcomes: $Y_i(1)$ (outcome under treatment) and $Y_i(0)$ (outcome under control). In the real world, we observe only the realized outcome:
$$Y_i = T_i Y_i(1) + (1 - T_i)Y_i(0)$$
We never observe the counterfactual potential outcome for the same individual. Thus, the individual-level treatment effect $\tau_i = Y_i(1) - Y_i(0)$ is fundamentally unobservable.
Because true $\tau_i$ does not exist as an observable label in real datasets like Criteo, metrics like **Mean Squared Error (MSE)** or **Precision in Estimating Heterogeneous Effects (PEHE)**:
$$\text{PEHE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{\tau}_i - \tau_i)^2}$$
**cannot be computed on real data**. Anyone claiming to report PEHE or causal ground-truth MSE on real customer data is mathematically incorrect. PEHE can only be computed in synthetic simulations where the true data-generating process (DGP) is known. On empirical data, evaluation must rely on ranking metrics (AUUC, Qini) and policy value estimators evaluated over randomized held-out samples.

---

### Q3: What are the mathematical formulations of the S-Learner, T-Learner, and X-Learner, and what are their respective failure modes?
**Answer:**
1. **S-Learner (Single Model)**:
   - Fits one model $\mu(X, T)$ predicting $Y$ on concatenated features $[X, T]$.
   - Predicts: $\hat{\tau}_S(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$.
   - *Failure Mode*: **Regularization Shrinkage Bias**. If features $X$ have strong predictive power on $Y$, tree algorithms or L2 regularization will split primarily on $X$ and penalize or drop the single indicator $T$, shrinking $\hat{\tau}(x)$ towards zero.
2. **T-Learner (Two Models)**:
   - Fits $\mu_1(X) \approx \mathbb{E}[Y \mid X, T=1]$ and $\mu_0(X) \approx \mathbb{E}[Y \mid X, T=0]$ independently.
   - Predicts: $\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$.
   - *Failure Mode*: **Arm Imbalance Variance**. In advertising, experiments are heavily imbalanced (e.g., Criteo has 85% treated, 15% control). The control model $\mu_0(X)$ is trained on vastly fewer samples and suffers from high variance, injecting noise into the subtracted difference.
3. **X-Learner (Künzel et al. 2019)**:
   - Stage 1: Fits $\mu_1(X)$ on treated, $\mu_0(X)$ on control.
   - Stage 2: Imputes individual treatment effects:
     $$D_1 = Y_1 - \hat{\mu}_0(X_1) \quad \text{for treated}$$
     $$D_0 = \hat{\mu}_1(X_0) - Y_0 \quad \text{for control}$$
   - Stage 3: Fits effect models $\hat{\tau}_1(X)$ on $D_1$, and $\hat{\tau}_0(X)$ on $D_0$.
   - Stage 4: Blends using propensity score $e(x)$:
     $$\hat{\tau}_X(x) = e(x)\hat{\tau}_0(x) + (1 - e(x))\hat{\tau}_1(x)$$

---

### Q4: Why does the X-Learner weight $\hat{\tau}_0(x)$ by $e(x)$ and $\hat{\tau}_1(x)$ by $(1 - e(x))$?
**Answer:**
This propensity weighting is an optimal variance-reduction technique:
When treatment assignment is heavily imbalanced with a large treated arm ($e(x) \approx 0.85$):
1. Model $\hat{\mu}_1(X)$ is estimated with very high precision because it has 85% of the data.
2. Therefore, the imputed effect $D_0 = \hat{\mu}_1(X_0) - Y_0$ for the minority control group has very low imputation noise.
3. This means that effect regressor $\hat{\tau}_0(X)$ is trained on clean, high-quality pseudo-effects! Thus, we assign it the largest weight: $e(x) \approx 0.85$.
4. Conversely, $\hat{\mu}_0(X)$ was trained on few control samples, so $D_1 = Y_1 - \hat{\mu}_0(X_1)$ is noisy. Thus, $\hat{\tau}_1(X)$ receives a lower weight: $(1 - e(x)) \approx 0.15$.
This directly targets and mitigates the primary weakness of the T-Learner.

---

### Q5: How does the Radcliffe Qini curve adjust for treatment-control imbalance?
**Answer:**
In randomized experiments where the treatment fraction is not 50/50, at any arbitrary ranking fraction $\phi$, the number of treated users $n_t(\phi)$ and control users $n_c(\phi)$ inside the top segment may fluctuate.
Radcliffe (2007) defined the Qini curve as:
$$Q(\phi) = \frac{Y_t(\phi) - Y_c(\phi) \cdot \frac{n_t(\phi)}{n_c(\phi)}}{N}$$
where:
- $Y_t(\phi)$ is the cumulative outcomes among treated units in top $\phi$.
- $Y_c(\phi)$ is the cumulative outcomes among control units in top $\phi$.
- The factor $\frac{n_t(\phi)}{n_c(\phi)}$ scales the control outcomes to match the size of the treated cohort inside that specific segment.
- Dividing by total sample size $N$ normalizes the curve so that values represent incremental outcomes per individual, making it comparable across sample sizes.

---

### Q6: Why is it crucial to distinguish between incremental visits per 1,000 targeted users vs per 1,000 total eligible users?
**Answer:**
Let $k$ be the targeting fraction (e.g. $k = 0.10$, the top 10% of users).
- **Incremental Visits per 1,000 Targeted Users**:
  $$\Delta \text{Visits}_{\text{targeted}} = (\bar{Y}_{t,\text{top } k} - \bar{Y}_{c,\text{top } k}) \times 1000$$
  This measures the efficiency of the campaign among the individuals who actually received the ad.
- **Incremental Visits per 1,000 Total Eligible Users**:
  $$\Delta \text{Visits}_{\text{eligible}} = (\bar{Y}_{t,\text{top } k} - \bar{Y}_{c,\text{top } k}) \times k \times 1000$$
  This measures the lift diluted across the entire addressable user base.
If a data scientist confuses these two in an executive presentation when $k=0.10$, they will exaggerate the expected commercial impact of targeting the entire population by a factor of 10!

---

### Q7: What is the Horvitz-Thompson IPW Policy Value estimator, and how does it evaluate targeting policies without confounding?
**Answer:**
Given any candidate targeting policy $\pi(X) \in \{0, 1\}$ (which decides whether to target user $X$):
The expected outcome under policy $\pi$ is $V(\pi) = \mathbb{E}[Y(\pi(X))]$.
Using Inverse Propensity Weighting (Horvitz & Thompson, 1952; Dudík et al., 2011), the unbiased estimator on randomized test data is:
$$\hat{V}(\pi) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{\mathbb{I}(T_i = 1, \pi(X_i) = 1)}{e(X_i)} Y_i + \frac{\mathbb{I}(T_i = 0, \pi(X_i) = 0)}{1 - e(X_i)} Y_i \right]$$
The term $\frac{\mathbb{I}(T_i = \pi(X_i))}{P(T_i \mid X_i)}$ reweights the observed outcomes of units whose randomized assignment happened to match the policy's recommendation.
The incremental policy value over status quo (treating nobody $\pi_0 \equiv 0$) simplifies to:
$$\Delta \hat{V}(\pi) = \hat{V}(\pi) - \hat{V}(\pi_0) = \frac{1}{N} \sum_{i: \pi(X_i)=1} \left[ \frac{T_i Y_i}{e_i} - \frac{(1 - T_i)Y_i}{1 - e_i} \right]$$
This evaluates the exact net volume of extra visits produced by the policy compared to zero marketing spend.

---

### Q8: Why do paired bootstrap confidence intervals resample both models on the exact same holdout units?
**Answer:**
If we bootstrapped Model A on one resample and Model B on an independent resample, the variance of the difference $\text{Var}(\hat{\Delta}) = \text{Var}(\hat{\theta}_A) + \text{Var}(\hat{\theta}_B)$ would include the sample-to-sample variance of the holdout data.
By evaluating both Model A and Model B on the **exact same bootstrap resample** $b$, we compute the paired difference:
$$\Delta^{(b)} = \text{Metric}_A(y^{(b)}, t^{(b)}, s_A^{(b)}) - \text{Metric}_B(y^{(b)}, t^{(b)}, s_B^{(b)})$$
This cancels out the shared holdout sample noise and isolates the true variance of the *difference* between the two model rankings:
$$\text{Var}(\hat{\theta}_A - \hat{\theta}_B) = \text{Var}(\hat{\theta}_A) + \text{Var}(\hat{\theta}_B) - 2\text{Cov}(\hat{\theta}_A, \hat{\theta}_B)$$
Since both models score the same customers, $\text{Cov}(\hat{\theta}_A, \hat{\theta}_B) > 0$, making the paired test substantially more statistically powerful.

---

### Q9: Why did you preserve duplicate feature rows instead of deleting them during data cleaning?
**Answer:**
In tabular machine learning, duplicate rows are often deleted under the assumption that they represent duplicate database entries. However, in large-scale anonymized ad tech logs:
1. Continuous features are coarse-grained, bucketed, or hashed for privacy.
2. Millions of distinct individuals share identical coarse feature combinations (e.g. same device category, broad geographic region, and browsing tier).
3. Deleting identical feature vectors would arbitrarily discard real user volume, distorting the empirical density of user segments and biasing treatment-control proportions.
Therefore, duplicates were audited, counted, and documented, but strictly preserved.

---

### Q10: Why is ad exposure (`exposure`) not a valid substitute for randomized treatment assignment (`treatment`)?
**Answer:**
In Criteo data, `treatment` denotes randomized ad *eligibility*, whereas `exposure` indicates whether an eligible user was actually served an ad (won the RTB auction, page loaded, ad rendered).
Conditioning on `exposure` introduces severe **collider bias / selection bias**:
1. Users who are exposed are systematically different from users who are not exposed (e.g. they visit websites with higher ad inventory, have faster internet connections, or higher engagement).
2. Users in the control group ($T=0$) can *never* be exposed (`exposure` = 0).
3. If we used `exposure` as treatment, the control group would no longer be comparable to the treated group, violating unconfoundedness $(Y(1), Y(0)) \perp\!\!\perp T$.
Therefore, `treatment` is the only valid randomized instrument.
