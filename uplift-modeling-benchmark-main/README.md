# uplift-modeling-benchmark

Documentation in Japanese: [README.ja.md](./README.ja.md)

Honest S/T/X meta-learner benchmark on the Criteo uplift dataset: bootstrap CIs on every metric, synthetic ground-truth validation, and explicit statements of what the data cannot conclude.

![Qini curves of the three learners with bootstrap confidence bands](assets/qini_criteo.png)

Uplift modeling answers "who should get the ad" rather than "who will convert": it estimates the individual treatment effect and targets the people whose behavior the ad actually changes. This project implements the S-, T- and X-learners from scratch (no causalml, no scikit-uplift), validates the whole harness on synthetic data where the true effect is known, and only then benchmarks the learners on one million rows of the Criteo randomized advertising experiment. Every reported number carries a bootstrap confidence interval, learner pairs are compared with paired-difference CIs on shared resamples, and where the intervals cannot separate two learners the notebook says so instead of declaring a winner.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | [Synthetic validation](notebooks/01_synthetic_validation.ipynb) | Plants a known treatment effect, then asserts that all three learners recover the true ATE on held-out data (max error 0.003 vs tolerance 0.005), that an oracle ranking beats random targeting on Qini, and that a random score scores approximately zero |
| 02 | [Criteo benchmark](notebooks/02_criteo_benchmark.ipynb) | EDA of the 13.9M-row experiment, a seeded 1M-row stratified subsample, S/T/X comparison with 95% bootstrap CIs on Qini, AUUC and uplift@10%, paired-difference CIs, Qini curves with bands and decile calibration |

Headline result of the committed run: all three learners beat random targeting (every Qini CI excludes zero), and none of them beats another (0 of 9 paired-difference CIs exclude zero across Qini, AUUC and uplift@10%). The X-learner has the highest point estimates, but this test set cannot rank the learners, and the notebook says exactly that instead of crowning a winner on noise.

## What "honest" means here

- Notebook 01 is a calibration of the measuring device. The metrics and learners run against a synthetic generator with a known per-row effect, and the checks are `assert` statements, not eyeballed plots. If the harness cannot recover a planted effect, real-data results are meaningless.
- All metrics on real data come with percentile bootstrap CIs (stratified by arm, seeded). Learners are compared on shared resamples so the differences are paired; overlapping marginal CIs are never used to claim a difference.
- The Limitations sections state what was not done: one split, one subsample, one base-model configuration, anonymized features, and a proxy outcome (`visit`, not profit).

## Quickstart

Python 3.11.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
jupyter lab
```

Notebook 01 runs without any download (about 20 seconds). Notebook 02 needs the raw Criteo file: the download is about 311 MB (a few minutes on a typical connection) and the notebook then runs in about 2 minutes; without the file it skips its computation cells with a message, which is also how it behaves in CI.

## Data and license

| Dataset | Source | License | Committed here |
|---|---|---|---|
| Criteo Uplift Prediction Dataset v2.1 | Hugging Face Hub, `criteo/criteo-uplift` (official Criteo AI Lab account) | released by Criteo for research use; CC BY-NC-SA 4.0 on the dataset card | nothing; download script only |
| Synthetic generator | `uplift_bench.synthetic.generate` | - | generated at run time |

Notes:

- The historical URL `go.criteo.net/criteo-research-uplift-v2.1.csv.gz`, still linked from the [Criteo AI Lab page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), returned 404 when this project was built (2026-07). `scripts/download_data.py` therefore fetches the official Hugging Face mirror first and keeps the historical URL as a fallback.
- The data is treated as not redistributable: `data/raw/` is gitignored and no real rows or aggregates are committed anywhere in this repository, including notebook outputs (which show only summary statistics of the public dataset).
- When using the dataset, cite Diemert, Betlei, Renaudin, Amini, "A Large Scale Benchmark for Uplift Modeling" (AdKDD 2018).

## Repository layout

```
notebooks/            executed notebooks (outputs committed)
notebooks_src/        jupytext py:percent sources of the same notebooks
src/uplift_bench/     learners, metrics, synthetic generator, data utils, plotting
tests/                pytest suite, including hand-computed metric checks
scripts/download_data.py
assets/               representative figure used above
```

## Limitations

- Single train/test split and a single 1M-row subsample; the bootstrap quantifies test-set noise, not split-to-split variation.
- The 12 features are anonymized, so no feature-level interpretation or business rule can be extracted.
- `visit` is a proxy outcome; an uplift model tuned for visits does not necessarily maximize conversions or profit, and the dataset contains no cost or revenue information.
- Intent-to-treat only; the `exposure` column is deliberately not used as a treatment indicator because exposure happens after randomization.
- One base-model configuration (regularized histogram gradient boosting), chosen for the rare outcome and validated on synthetic data; learner rankings can change with the base model.

## License

Code is MIT licensed (see [LICENSE](./LICENSE)). The Criteo dataset has its own terms (table above) and is not part of this repository.
