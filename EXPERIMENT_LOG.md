# MaintenAI – Experiment Log

Every model experiment is recorded here with its configuration and **measured** results. No model has been trained
yet, so there are no results.

| ID | Date | Milestone | Experiment | Dataset (MD5) | Split | Features | Config | Results | Notes |
|---|---|---|---|---|---|---|---|---|---|
| EDA-01 | 2026-11-17 | 01 | Dataset inspection and EDA (no model) | ai4i2020.csv (`f0adaa9c…`) | stratified 70/15/15, seed 42 (planned) | n/a | `notebooks/make_01_eda.py` | See `reports/eda_summary.json` | Rule agreement for HDF/PWF/OSF = 100% |

## Template for model runs

```
ID:            EXP-<n>
Date:
Milestone:
Question:      (RQ1–RQ6)
Model(s):
Feature set:   raw | engineered
Imbalance:     none | class_weight | oversample | SMOTE | threshold
Split / CV:    e.g. repeated stratified 5-fold x3 on train; validation for thresholds
Hyperparameters:
Seed:
Metrics (mean ± sd over CV folds, and validation):
  PR-AUC, ROC-AUC, failure recall, failure precision, failure F1, macro F1, accuracy, Brier, ECE
Train time / inference time / model size:
Artifacts:     artifacts/models/..., artifacts/metrics/..., MLflow run id
Observations:
```

## 17 Nov 2026 – Experiments 1–8 (single stratified split, seed 42; test = 1,500 rows, 51 failures)
Command: `python -m ml.experiments.run_all`. All values from `artifacts/metrics/results.json`.
- Exp 1 test PR-AUC raw: CART 0.721, AdaBoost 0.706, GB 0.838, XGB 0.866, MARS 0.704, GAM 0.666, RF 0.852. Engineered: CART 0.920, Ada 0.947, GB 0.940, XGB 0.931, MARS 0.843, GAM 0.757, RF 0.931.
- Exp 2 raw: Voting 0.870, Stacking 0.878, BMA 0.838 (weight 1.000 on GB, both feature sets – Minka caveat). Engineered: 0.937 / 0.938 / 0.940.
- Exp 3 XGB raw: threshold tuning ≥ resampling; SMOTE lowest PR-AUC (0.819).
- Exp 5: base models already calibrated (ECE ≤ 0.012); Platt on validation worsened ECE.
- Exp 6: deferring 10% most uncertain (entropy) → error 1.40% → 0.22%, 40/51 failures routed to review.
- Exp 7 ablation and Exp 8 leave-one-type-out: see results.json; held-out type H raw XGB 0.445.
- Exp 4: engineered CART recovers documented thresholds (10,998.5 min·Nm; 3,492.8 / 9,004 W; 8.65 K; 1,380.5 rpm).
Limitations: one split; wide bootstrap CIs; only CART/XGB tuned.
