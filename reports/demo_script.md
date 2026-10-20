# IncrementalImpact: 5-Minute Portfolio Presentation & Demo Script

**Target Audience:** Senior Data Scientists, Machine Learning Hiring Managers, Technical Interviewers  
**Speaker:** Lakshya Arora  
**Topic:** Causal Uplift Modelling vs Traditional Response Prediction in Digital Advertising  

---

## Timeline & Presentation Cues

| Time | Phase | Focus & Screen Cue | Key Message |
| :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | **The Problem** | Show Streamlit Page 1 (2x2 Matrix) | Why predicting response wastes ad spend on Always-Takers. |
| **1:00 - 2:00** | **The Methodology** | Show Streamlit Page 2 & Code Snippets | The Fundamental Problem of Causal Inference & S/T/X Meta-Learners. |
| **2:00 - 3:00** | **The Benchmark** | Show Streamlit Page 3 (Leaderboard) & Page 4 (Curves) | Radcliffe Qini, AUUC, and Paired Bootstrap CIs. |
| **3:00 - 4:00** | **Live Interactive Demo** | Show Streamlit Page 5 (Simulator) & FastAPI `/predict` | What-if financial budget simulation under explicit unit economics. |
| **4:00 - 5:00** | **Conclusion & Defense** | Show Streamlit Page 6 & GitHub Repository | Engineering rigor, CI test suite (33 passing tests), and limitations. |

---

## Detailed Spoken Script

### Minute 0:00 - 1:00: The Problem (Hook & Business Motivation)
> *"Hello! Today I’m presenting **IncrementalImpact**, an end-to-end causal machine learning system built for smarter advertising decision-making.*
>
> *In commercial ad tech, companies spend millions predicting 'Who is going to click or visit?' using supervised response classifiers. But that formulation has a catastrophic blind spot: **it targets Always-Takers**—customers who already intended to visit organically.*
>
> *On screen, you see the classic causal customer quadrant: Persuadables, Always-Takers, Lost Causes, and Do-Not-Disturbs. A standard response model targets users with the highest probability $P(Y=1 \mid T=1)$, which heavily selects Always-Takers. What we actually want to optimize is the **incremental causal uplift**: $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]$. We only want to spend advertising budget on Persuadables."*

---

### Minute 1:00 - 2:00: Methodology & The Fundamental Problem
> *"Now, why is this challenging? Because of the **Fundamental Problem of Causal Inference**: for any individual user, we only observe one realized outcome. We never observe both potential outcomes simultaneously. Therefore, we cannot compute Mean Squared Error or supervised loss against individual ground truth.*
>
> *To solve this, I implemented three meta-learner architectures from the literature: the **S-Learner**, the **T-Learner**, and the **X-Learner** (Künzel et al. 2019).*
>
> *The X-Learner is particularly suited for ad tech because our dataset—the Criteo Uplift Benchmark—has an **85% treated to 15% control imbalance**. The X-Learner estimates counterfactuals in stage one, fits effect models in stage two, and blends them using propensity scores $e(x)$, weighting the lower-variance model more heavily."*

---

### Minute 2:00 - 3:00: Benchmark Results & Evaluation Rigor
> *(Transition to Streamlit Page 3 & Page 4)*
>
> *"Here on the dashboard is our held-out test leaderboard across 10,001 untouched units.*
>
> *Notice our findings: The conventional **Response Model achieved a negative AUUC (-0.0013) and negative Qini (-0.0011)**. Even though it had high conversion rates, it failed to deliver incremental lift above random targeting.*
>
> *Meanwhile, our **X-Learner achieved the highest overall AUUC (+0.0018) and Qini (+0.0016)**, driving an **incremental uplift of +4.26% at the top 10% targeted**, or 42.6 incremental visits per 1,000 targeted users.*
>
> *Crucially, I didn’t just report point estimates. I implemented **paired bootstrap confidence intervals** where both models are evaluated on the exact same resampled test units. This ensures that any observed superiority is subjected to rigorous statistical scrutiny."*

---

### Minute 3:00 - 4:00: Live Interactive Demo & Budget Simulator
> *(Transition to Streamlit Page 5 & FastAPI terminal)*
>
> *"To bridge the gap between academic CATE estimation and marketing operations, I built this **What-If Budget Simulator**.*
>
> *Let's set a hypothetical campaign budget of **$2,500**, an ad delivery cost of **$0.50 per targeted user**, and an estimated gross value of **$25.00 per incremental visit**.*
>
> *The system computes the economic break-even uplift threshold: $\tau^* = \frac{0.50}{25.00} = 2.0\%$. Under the exact same spend cap, our Uplift Model targets strictly profitable customers, generating **$4,250 in gross revenue and $1,750 in net incremental profit (1.70x ROAS)**, whereas the response model dilutes budget on organic visitors.*
>
> *The entire inference engine is backed by a production-ready **FastAPI microservice**, complete with Pydantic validation, health checks, and batch scoring up to 10,000 requests."*

---

### Minute 4:00 - 5:00: Engineering Rigor, Limitations & Conclusion
> *(Transition to Streamlit Page 6 & Codebase)*
>
> *"To ensure production-grade reproducibility:
> 1. The repository features **33 comprehensive automated tests** covering data leakage, non-binary validation, and synthetic null tests where true effect is zero.
> 2. Experiment metadata and metrics are tracked locally via **MLflow and DuckDB**.
> 3. I documented critical failure modes: `exposure` is not randomized treatment assignment, and identical anonymized feature vectors are preserved to avoid distorting density.*
>
> *In summary, IncrementalImpact demonstrates how causal inference can transform marketing efficiency from vanity conversion metrics into measurable, defendable business value.*
>
> *Thank you, and I look forward to your questions!"*
