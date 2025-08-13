# CTGAN vs TVAE on Adult — Reproduction Report

**Author:** Oscar Nolen

**Date:** 08/13/2025

---

## 1) Setup

- **Repo used:** sdv-dev/CTGAN (CTGAN + TVAE)
- **Dataset:** Adult (UCI)
- **Split:** Deterministic stratified split (≈23k train / 10k test target; exact sizes are in `outputs/config.json`)
- **Hyperparameters:** Epochs=300, Batch size=500 for both CTGAN and TVAE (per paper benchmark setup)
- **Seed:** 42
- **Environment:** Python 3.12, scikit-learn, ctgan, sdv (see config.json for exact versions)

**Citations**
- Paper benchmark uses 300 epochs and batch size 500 for all models.  
- Machine learning efficacy protocol: train on synthetic, **test on real test set**.

---

## 2) Task 2 — Machine Learning Efficacy (Synthetic → Real Test)

We train four classifiers on **synthetic** training data and evaluate on the **real** test set.

Classifiers:
- AdaBoost (n_estimators=50)
- Decision Tree (max_depth=20)
- Logistic Regression
- MLP (hidden_layer_sizes=(50,))

**Results (from `metrics_task2.csv`):**

| Source | Classifier | Test Accuracy | Test F1 |
|---|---|---:|---:|
| CTGAN_synth_train | AdaBoost(n=50) | 0.834 | 0.596 |
| CTGAN_synth_train | DecisionTree(max_depth=20) | 0.785 | 0.550 |
| CTGAN_synth_train | LogisticRegression | 0.842 | 0.654 |
| CTGAN_synth_train | MLP(50) | 0.809 | 0.589 |
| TVAE_synth_train | AdaBoost(n=50) | 0.845 | 0.657 |
| TVAE_synth_train | DecisionTree(max_depth=20) | 0.796 | 0.577 |
| TVAE_synth_train | LogisticRegression | 0.840 | 0.659 |
| TVAE_synth_train | MLP(50) | 0.827 | 0.643 |

**Aggregate view (accuracy):**

| classifier                 |   CTGAN_synth_train |   TVAE_synth_train |
|:---------------------------|--------------------:|-------------------:|
| AdaBoost(n=50)             |               0.834 |              0.845 |
| DecisionTree(max_depth=20) |               0.785 |              0.796 |
| LogisticRegression         |               0.842 |              0.840 |
| MLP(50)                    |               0.809 |              0.827 |

**Aggregate view (F1):**

| classifier                 |   CTGAN_synth_train |   TVAE_synth_train |
|:---------------------------|--------------------:|-------------------:|
| AdaBoost(n=50)             |               0.596 |              0.657 |
| DecisionTree(max_depth=20) |               0.550 |              0.577 |
| LogisticRegression         |               0.654 |              0.659 |
| MLP(50)                    |               0.589 |              0.643 |

**Are your results the same as reported in the paper? Why/why not?**  
- Differences are common due to preprocessing variants, library version drift, split differences, and optimizer stochasticity.
- Diagnostics to validate:
  - Confirm discrete column list and label encoding.
  - Check that synthetic datasets include the **label** `income` and preserve class balance.
  - Compare marginals and correlations (e.g., SDMetrics).
  - Adjust CTGAN/TVAE epochs (e.g., 200↔400) and re-run to see stability.

---

## 3) Task 3 — Real → Real Test Baseline

Train the same classifiers on **real train** and evaluate on the **real test**.

**Results (from `metrics_task3.csv`):**

| Source | Classifier | Test Accuracy | Test F1 |
|---|---|---:|---:|
| REAL_train | AdaBoost(n=50) | 0.851 | 0.647 |
| REAL_train | DecisionTree(max_depth=20) | 0.828 | 0.638 |
| REAL_train | LogisticRegression | 0.850 | 0.665 |
| REAL_train | MLP(50) | 0.841 | 0.656 |

**Utility gap (Task 3 – Task 2 averages)**:  
- Accuracy gap: 0.020  
- F1 gap: 0.036

**Interpretation:** A positive gap indicates synthetic data still trails real data (typical). If negative, investigate for leakage or split mismatches.

---

## 4) Task 4 — Privacy via Mean DCR

Definition (TabDDPM): For each synthetic sample, compute the **minimum L2 distance** to the set of **real training** records; the **mean** over synthetic samples is the **mean DCR**.

**Results (from `dcr_results.csv`):**

| model   |   size_multiple |   mean_DCR |
|:--------|----------------:|-----------:|
| CTGAN   |               1 |     1.5323 |
| CTGAN   |               2 |     1.5303 |
| CTGAN   |               4 |     1.5323 |
| TVAE    |               1 |     0.9252 |
| TVAE    |               2 |     0.9285 |
| TVAE    |               4 |     0.9276 |

![DCR vs Synthetic Size](dcr_trend.png)

**Trend & Interpretation:**  
- Expectation: mean DCR should **not** decrease as synthetic size increases; drops can indicate duplicates/mode collapse.  
- Interpret DCR alongside Task 2 metrics; very high DCR with poor Task 2 results may indicate OOD samples (privacy without utility).

---

## 5) Reproducibility & Next Steps

- Exact configuration saved to `outputs/config.json`.
- Suggested validations:
  - Re-run with multiple seeds and average.
  - Add SDMetrics (e.g., CSTest, TVD) for distribution fidelity.
  - Consider SDGym splits to match the paper precisely.
  - Try CTGAN’s conditional sampling to balance rare categories for augmentation.

---

## 6) References

- Xu et al., **Modeling Tabular Data using Conditional GAN (CTGAN)**, NeurIPS 2019.  
- Kotelnikov et al., **TabDDPM: Modelling Tabular Data with Diffusion Models**, ICML 2023 (DCR definition).