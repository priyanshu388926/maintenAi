# MaintenAI – Project Status

**Current milestone:** ML engine and experiments 1–8 complete (milestones 02–05 run together at the team's request for the course report; review gates skipped)
**Web application:** built – FastAPI back end (`backend/`), React + TypeScript front end (`frontend/`), MongoDB or file storage, Docker files; start with `bash run_web.sh`
**Next:** validation on real run-to-failure data; user testing of the web app
**Last updated:** 17 Nov 2026

## Milestone 01 acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Research foundation established | ✅ (initial pass) | `RESEARCH_NOTES.md` §1–2: RQs, ~25 references, candidate gaps. Systematic review still to do |
| AI4I dataset inspected | ✅ | `data/raw/ai4i2020.csv` (mirror copy, MD5 recorded), `notebooks/01_dataset_exploration.ipynb` executed |
| Dataset schema documented | ✅ | `RESEARCH_NOTES.md` §3.2, `ml/data/schema.py` |
| EDA completed | ✅ | Notebook 01, `reports/eda_summary.json`, 4 figures in `reports/figures/` |
| Leakage-safe preprocessing designed | ✅ | `RESEARCH_NOTES.md` §4–5, `assert_no_leakage` + tests |

## Completed
- Preprocessing pipeline, 7 base models (incl. own MARS), voting, stacking, BMA: `ml/`. 21 tests pass.
- All experiments run via `python -m ml.experiments.run_all` (285 s); results in `artifacts/metrics/results.json`, figures in `reports/figures/`.
- Course report: `21CSC305P_MaintenAI_Report.docx` (20 pages).
- Project skeleton created (only folders that are used).
- Dataset obtained, integrity-checked, validated against the documented schema.
- Data layer: `ml/data/schema.py`, `ml/data/validation.py`, `ml/data/ingestion.py`.
- 14 pytest tests passing (`python -m pytest`).
- EDA notebook executed end to end; summary JSON and figures saved.
- Generating rules of HDF / PWF / OSF verified against the data (100% agreement).
- Split plan fixed and checked (7,000 / 1,500 / 1,500 rows; 237 / 51 / 51 failures).
- Dependency probe: pyGAM, XGBoost, SHAP OK; py-earth (MARS) does not build.

## Pending / open issues
1. Dataset came from GitHub mirrors because the UCI site was blocked in the development environment. Download the
   official file and compare MD5 `f0adaa9c5e25366c6df1cc2684479871`.
2. Systematic literature search (IEEE Xplore, ScienceDirect, Scopus/Google Scholar) to confirm or reject the
   candidate gaps; verify page numbers/DOIs of references marked ○.
3. Choose the MARS implementation (Milestone 3).
4. NASA IMS dataset not yet investigated.
5. Python 3.11 environment not yet created (probe ran on 3.13).

## Files created in Milestone 01
```
README.md  PROJECT_STATUS.md  EXPERIMENT_LOG.md  RESEARCH_NOTES.md  requirements.txt  pytest.ini  .gitignore
data/raw/ai4i2020.csv                (not tracked by Git)
ml/__init__.py  ml/data/__init__.py  ml/data/schema.py  ml/data/validation.py  ml/data/ingestion.py
tests/test_data.py
notebooks/make_01_eda.py  notebooks/01_dataset_exploration.ipynb
reports/eda_summary.json  reports/figures/eda_*.png (4 files)
```

## Decisions
- Primary metric: PR-AUC and failure recall at a validation-chosen precision; accuracy reported but not headline.
- Two feature sets in every experiment: raw and engineered (because engineered features reproduce the simulator's rules).
- Test set locked until final evaluation.
- Official `Machine failure` label kept unchanged; its inconsistencies are documented.
- Failure-mode task treated as multi-label; RNF excluded as a prediction target (random by construction).

## Original Milestone 02 plan (completed)
1. Create a Python 3.11 virtual environment and install `requirements.txt`.
2. `ml/data/preprocessing.py`: split function (stratified 70/15/15, seed 42) saving split indices to
   `data/processed/splits.json`; `ColumnTransformer` builders for the raw and engineered feature sets.
3. `ml/data/feature_engineering.py`: the three derived features as a scikit-learn transformer.
4. `ml/models/cart.py`: CART with Gini vs entropy, depth, min samples, cost-complexity pruning (`ccp_alpha`) chosen
   by repeated stratified CV on PR-AUC.
5. `ml/evaluation/metrics.py`: metric suite (PR-AUC, ROC-AUC, recall, F1, Brier, ECE, confusion matrix).
6. Notebooks 02 and 03; tree visualisation and rule extraction; first entries in `EXPERIMENT_LOG.md`.
7. Tests for split stratification, transformer fitting on train only, and CART probability validity.
