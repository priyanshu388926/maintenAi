# MaintenAI – Research Notes

Working notes for the research side of the project: questions, literature, dataset facts, methodological
decisions and open risks. Updated at every milestone. Last update: Milestone 01 (17 Nov 2026).

---

## 1. Research questions (not yet answered)

| ID | Question | Planned evidence |
|---|---|---|
| RQ1 | How well do individual CART, Boosting, GAM and adaptive basis function models predict machine failure? | Experiment 1 |
| RQ2 | Does ensemble learning improve failure detection over individual classifiers? | Experiment 2, 7 |
| RQ3 | Does Bayesian Model Averaging improve reliability when models have different strengths? | Experiments 2, 5, 7 |
| RQ4 | Can interpretable models identify the variables that drive failure risk? | Experiment 4 |
| RQ5 | Can uncertainty estimates flag unreliable predictions for human inspection? | Experiment 6 |
| RQ6 | How well does the framework generalise across operating conditions / independent data? | Experiment 8 |

No result for any RQ exists yet. Nothing in this file should be read as a finding about model performance.

---

## 2. Literature review (initial pass)

**Status.** This is a first pass from web searches and established textbook/journal knowledge, not yet a
systematic review. Entries marked ✔ were checked against a publisher or index page during Milestone 01; entries
marked ○ are well-known works whose bibliographic details (pages, DOI) still need checking in Google Scholar /
IEEE Xplore / ScienceDirect before they go into the paper.

### 2.1 Predictive maintenance (PdM) background

| Ref | Work | Relevance |
|---|---|---|
| ✔ | T. P. Carvalho et al., "A systematic literature review of machine learning methods applied to predictive maintenance," *Computers & Industrial Engineering*, vol. 137, art. 106024, 2019. | Survey of ML methods used in PdM; random forests, neural networks, SVMs and k-means are among the most frequent. Baseline for the related-work section. |
| ○ | T. Zonta et al., "Predictive maintenance in the Industry 4.0: A systematic literature review," *Computers & Industrial Engineering*, vol. 150, art. 106889, 2020. | Second systematic review; taxonomy of PdM approaches. |
| ○ | J. Lee et al., "Prognostics and health management design for rotary machinery systems—Reviews, methodology and applications," *Mechanical Systems and Signal Processing*, vol. 42, pp. 314–334, 2014. | PHM methodology; context for the NASA IMS bearing extension. |
| ✔ | S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," *Proc. 3rd Int. Conf. Artificial Intelligence for Industries (AI4I)*, IEEE, 2020 (IEEE Xplore document 9253083). | **Origin of the AI4I 2020 dataset.** Mandatory citation; describes the synthetic generator and failure-mode rules. |
| ○ | H. Qiu, J. Lee, J. Lin, G. Yu, "Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics," *Journal of Sound and Vibration*, vol. 289, pp. 1066–1090, 2006. | Customary citation for the NASA/IMS bearing dataset (secondary dataset). |
| ○ | A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," *Proc. Int. Conf. Prognostics and Health Management*, 2008. | C-MAPSS origin (optional RUL extension only). |

### 2.2 Unit 5 algorithms

| Ref | Work | Used for |
|---|---|---|
| ○ | L. Breiman, J. Friedman, R. Olshen, C. Stone, *Classification and Regression Trees*. Wadsworth, 1984. | CART |
| ○ | Y. Freund, R. E. Schapire, "A decision-theoretic generalization of on-line learning and an application to boosting," *J. Computer and System Sciences*, vol. 55, no. 1, pp. 119–139, 1997. | AdaBoost |
| ○ | J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001. | Gradient boosting |
| ○ | T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system," *Proc. ACM SIGKDD*, pp. 785–794, 2016. | XGBoost |
| ○ | J. H. Friedman, "Multivariate adaptive regression splines," *Annals of Statistics*, vol. 19, no. 1, pp. 1–67, 1991. | MARS (adaptive basis functions) |
| ○ | T. Hastie, R. Tibshirani, *Generalized Additive Models*. Chapman & Hall, 1990. | GAM |
| ○ | L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. | Random forest |
| ○ | D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992. | Stacking |
| ✔ | J. A. Hoeting, D. Madigan, A. E. Raftery, C. T. Volinsky, "Bayesian model averaging: A tutorial," *Statistical Science*, vol. 14, no. 4, pp. 382–417, 1999. | BMA definition and practice |
| ○ | A. E. Raftery, "Bayesian model selection in social research," *Sociological Methodology*, vol. 25, pp. 111–163, 1995. | BIC approximation to posterior model probabilities |
| ○ | T. P. Minka, "Bayesian model averaging is not model combination," MIT Media Lab technical note, 2000. | **Key caveat for RQ3** (see §5.4) |
| ○ | Y. Yao, A. Vehtari, D. Simpson, A. Gelman, "Using stacking to average Bayesian predictive distributions," *Bayesian Analysis*, vol. 13, no. 3, pp. 917–1007, 2018. | Alternative to BMA when no candidate model is "true" |
| ○ | K. P. Murphy, *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012 (Ch. 16 "Adaptive basis function models"). | Course textbook framing of Unit 5 |

### 2.3 Imbalance, calibration, explainability, uncertainty

| Ref | Work | Used for |
|---|---|---|
| ○ | N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," *JAIR*, vol. 16, pp. 321–357, 2002. | Experiment 3 |
| ○ | A. Niculescu-Mizil, R. Caruana, "Predicting good probabilities with supervised learning," *Proc. ICML*, 2005. | Calibration of boosted trees / RF (Experiment 5) |
| ○ | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural networks," *Proc. ICML*, 2017. | Expected Calibration Error definition |
| ○ | S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017. | SHAP (Experiment 4) |
| ○ | B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," *NeurIPS*, 2017. | Ensemble disagreement as uncertainty (Experiment 6) |

### 2.4 What existing AI4I work already covers

A quick search shows many public projects on AI4I that combine random forest / XGBoost with SHAP, class
weighting or SMOTE, PR-AUC and probability calibration (e.g. GitHub repositories by Edirin-Ako, Yvngval,
syedyousafshah997, subhamsahu1995, dipenpatel1), and at least one study of SHAP stability on AI4I (Warwick URSS
showcase). **Therefore "ensemble + SHAP + calibration on AI4I" is not by itself a research gap.**

### 2.5 Candidate gaps (to be confirmed by a systematic search in Week 1–2, not claimed yet)

1. **Rule-recovery transparency.** AI4I failure labels are produced by deterministic thresholds on derived
   physical quantities (verified in our EDA, §3.4). Many reported near-perfect scores may reflect recovery of these
   rules. A study that separates *raw-feature* from *rule-aligned-feature* performance and checks whether
   interpretable models (CART, GAM, MARS) recover the true generating thresholds would be a useful, honest
   contribution.
2. **BMA across heterogeneous model families with explicit posterior weights**, compared against stacking and
   soft voting on the same splits, including how the posterior concentrates (Minka's caveat). Not yet found in the
   AI4I literature during this first pass.
3. **Selective prediction for maintenance.** Using calibrated probability, entropy and model disagreement to defer
   uncertain cases to a human, reported as risk–coverage curves.
4. **Operating-condition shift.** Train on L/M product types, test on H (and permutations) as a proxy for
   generalisation, since machine-grouped splits are impossible (§3.2).

IEMAF is treated as a **combination framework of established methods**, not as a new algorithm.

---

## 3. Dataset: UCI AI4I 2020 Predictive Maintenance

### 3.1 Source and provenance

- Official page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset (also published
  on Kaggle by the author, S. Matzka). Licence on UCI: CC BY 4.0 (to confirm on the page when it is reachable).
- **Access in Milestone 01:** the UCI site was blocked by the network proxy of the development environment (HTTP 403
  from both the cloud workspace and the local VM). The file was instead obtained from two independent public GitHub
  repositories (`ALP-gthub/ai4i-predictive-maintenance` and `Edoar-do/AI4I-Predictive-Maintenance`). Both copies are
  byte-identical: MD5 `f0adaa9c5e25366c6df1cc2684479871`, 522,048 bytes, 10,000 data rows. Shape, class counts and
  failure-mode counts match the official documentation.
- **Action:** download the official file from UCI on a normal network and compare its MD5. If it differs, replace
  the file and re-run the notebook.
- The file starts with a UTF-8 byte-order mark; `ml/data/ingestion.py` reads it with `encoding="utf-8-sig"`.
- **The data are synthetic.** They were generated to mirror real industrial behaviour, not recorded from real
  machines. Every claim in the paper must say so.

### 3.2 Data dictionary

| Column | Role | Type | Range observed | Meaning |
|---|---|---|---|---|
| UDI | identifier (excluded) | int | 1–10,000 | Row index |
| Product ID | identifier (excluded) | string | unique per row | Quality letter + serial number. Unique for every row, so there are **no repeated machines** |
| Type | input, categorical | L / M / H | 6,000 / 2,997 / 1,003 | Product quality variant (low / medium / high); equals the first letter of Product ID |
| Air temperature [K] | input | float | 295.3–304.5 | Ambient temperature (random walk, σ ≈ 2 K) |
| Process temperature [K] | input | float | 305.7–313.8 | Process temperature (≈ air + 10 K); correlation with air temperature 0.876 |
| Rotational speed [rpm] | input | int | 1,168–2,886 | Spindle speed |
| Torque [Nm] | input | float | 3.8–76.6 | Torque; correlation with speed −0.875 |
| Tool wear [min] | input | int | 0–253 | Accumulated tool use |
| Machine failure | **target** | 0/1 | 339 positives (3.39%) | Machine failed in this data point |
| TWF | failure-mode label | 0/1 | 46 | Tool wear failure |
| HDF | failure-mode label | 0/1 | 115 | Heat dissipation failure |
| PWF | failure-mode label | 0/1 | 95 | Power failure |
| OSF | failure-mode label | 0/1 | 98 | Overstrain failure |
| RNF | failure-mode label | 0/1 | 19 | Random failure (0.1% chance regardless of conditions) |

No missing values, no duplicate rows, no out-of-range values.

### 3.3 Label facts that affect modelling

- Class imbalance: 9,661 normal vs 339 failures. Majority-class accuracy is **96.61%**, so accuracy is not a useful
  headline metric.
- Failure rate by type: L 3.92%, M 2.77%, H 2.09%.
- 9 rows have `Machine failure = 1` with no mode flag.
- 18 of the 19 RNF rows have `Machine failure = 0`; only 1 RNF row is labelled as a machine failure.
- 24 rows carry more than one mode flag (mostly PWF + OSF), so failure-mode classification is **multi-label**.
- We keep the official `Machine failure` label unchanged and document these inconsistencies.

### 3.4 Generating rules (verified, 100% agreement on all 10,000 rows)

| Mode | Rule (from the dataset documentation) | Agreement in data |
|---|---|---|
| HDF | process − air temperature < 8.6 K **and** speed < 1,380 rpm | 115/115, no false positives |
| PWF | power P = τ·ω < 3,500 W or > 9,000 W | 95/95, no false positives |
| OSF | tool wear × torque > 11,000 / 12,000 / 13,000 min·Nm for L / M / H | 98/98, no false positives |
| TWF | tool replaced/failed at a random wear between ~200 and 240 min | observed wear at TWF: 198–253 min (median 214.5) |
| RNF | random, 0.1% | not predictable from inputs by design |

Consequences:
- Three engineered features (temperature difference, mechanical power, wear × torque) are physically justified and
  will make HDF, PWF and OSF almost perfectly separable. They are also **exactly the variables the simulator used**,
  so performance with them measures rule recovery on a simulator, not real-world predictive power.
- Every experiment will therefore be run on two feature sets: **raw** (Type + 5 operating variables) and
  **engineered** (raw + 3 derived quantities). This doubles as part of the ablation.
- The ceiling for Machine failure is limited by TWF timing randomness, RNF and the 9 unexplained failures; a recall
  of 100% should not be expected.

---

## 4. Leakage rules

| Target | Forbidden inputs | Reason |
|---|---|---|
| Machine failure | UDI, Product ID, TWF, HDF, PWF, OSF, RNF | Mode flags are sub-labels of the target (any of TWF–OSF = 1 implies failure = 1 in this data); identifiers carry no physical meaning and Product ID encodes Type |
| One failure mode (e.g. HDF) | UDI, Product ID, Machine failure, all other modes | All labels are derived from the same events |

Enforced in code by `ml.data.validation.assert_no_leakage`, covered by `tests/test_data.py`. Every model-building
function in later milestones must call it.

Other leakage controls:
- All preprocessing (scaling, encoding, imputation if ever needed, resampling) is fitted on the training fold only,
  inside a scikit-learn `Pipeline` / `ColumnTransformer`.
- SMOTE and random oversampling run **inside** cross-validation folds (imblearn `Pipeline`), never before splitting.
- Calibration, BMA weights that use held-out likelihood, stacking meta-learners and risk thresholds are fitted on
  validation data or out-of-fold predictions, never on the test set.
- AI4I row order is an index, not time. No lag or rolling features; no chronological claims.

---

## 5. Methodological decisions

### 5.1 Splits
- Stratified 70 / 15 / 15 train / validation / test split, seed 42 → 7,000 / 1,500 / 1,500 rows with 237 / 51 / 51
  failures.
- The test set is locked: touched once, at the end of Milestone 5 / 7, for final numbers.
- Model selection: repeated stratified 5-fold cross-validation (3 repeats) on the training set.
- Uncertainty in metrics: bootstrap 95% confidence intervals on the test set (51 failures is small).
- Generalisation proxy: leave-one-Type-out (train on two product types, test on the third).

### 5.2 Preprocessing plan
- Type: one-hot encoding for linear/GAM/MARS models; ordinal (L < M < H) also acceptable for trees.
- Numeric features: standardisation for models that need it (GAM, MARS, RBF, logistic meta-learner); trees use raw
  values so that CART thresholds stay in physical units.
- Derived features (engineered set only): `temp_diff_K`, `power_W`, `wear_torque_minNm`.
- Imbalance strategies compared in Experiment 3: none, class weights, random oversampling, SMOTE, threshold tuning.

### 5.3 Metrics
- Primary: **PR-AUC (average precision)** and **failure-class recall at a validation-chosen precision**; failure-class
  F1; ROC-AUC.
- Also: accuracy, macro F1, confusion matrix, false negative / false positive rate.
- Probability quality: Brier score, Expected Calibration Error (10 equal-width bins), reliability diagrams.
- Cost: training time, inference time, model size.

### 5.4 Bayesian Model Averaging – plan and caveat
- Planned approximation: posterior model probability P(M_m | D) ∝ P(M_m) · exp(−BIC_m / 2), or the validation
  log-likelihood under an explicit likelihood model, with a uniform prior over models. Clearly labelled as an
  approximation.
- Problem: BIC needs a parameter count, which is ill-defined for trees and boosted ensembles. Options to evaluate in
  Milestone 4: (a) held-out log predictive density as an approximation to the log evidence; (b) restricting exact
  BIC to models with well-defined parameters (GAM effective degrees of freedom, MARS terms, logistic models).
- **Caveat (Minka 2000):** BMA assumes one of the candidate models is true; as data grow, posterior weights
  concentrate on a single model. BMA may therefore behave like model selection rather than combination. We will
  report the posterior weights explicitly and compare with stacking of predictive distributions (Yao et al. 2018).

---

## 6. Dependency risks found in Milestone 01

| Package | Status | Plan |
|---|---|---|
| pyGAM 0.12.0 | Installs and fits a `LogisticGAM` (tested with numpy 2.x, scipy 1.16) | Use for GAM |
| XGBoost 3.x, SHAP 0.5x | Install and import | Use |
| sklearn-contrib-py-earth (MARS) | **Fails to build** on a modern Python | Milestone 3: evaluate (1) a maintained fork, (2) R `earth` via rpy2, (3) a documented NumPy implementation of Friedman's forward/backward algorithm with GCV pruning. Not replacing MARS with an unrelated model |

The probe used Python 3.13; the project targets Python 3.11 and will be re-checked in that environment.

---

## 7. Secondary datasets (status)

- NASA IMS bearings: not yet downloaded or inspected. Different schema (raw vibration signals, run-to-failure
  experiments) – needs its own adapter. Optional extension.
- NASA C-MAPSS: not used for the main project (RUL is out of scope unless time allows).
