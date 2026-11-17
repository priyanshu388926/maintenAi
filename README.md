# MaintenAI – Intelligent Predictive Industrial Maintenance System

Research prototype for predicting industrial machine failure with interpretable, ensemble and Bayesian model
averaging methods (Unit 5 of 21CSC305P Machine Learning). Working framework name: **IEMAF – Interpretable Ensemble
Model Averaging Framework**, a combination of established methods rather than a new algorithm.

> Status: ML engine, experiments 1–8 and the web application (FastAPI + React + MongoDB/file storage) are built.
> See `PROJECT_STATUS.md`.

## Run the web application (quick start)

```bash
cd ~/Documents/ML_Course_Projects/MaintenAI
bash run_web.sh          # first run: installs packages and trains the models (~5 min), then opens http://localhost:8000
```

Stop it with Ctrl+C. Next time `bash run_web.sh` starts in a few seconds. API documentation: http://localhost:8000/docs.

| Page | What it does |
|---|---|
| Dashboard | Dataset KPIs, failure modes, failure rate by type, live model quality, logged risk levels, recent predictions |
| Prediction Centre | Enter readings (or presets) → probability from 7 models, voting, stacking, BMA and calibrated BMA (IEMAF); risk level Low / Medium / High / Uncertain; review flag; SHAP contributions; CART decision path; documented failure-rule checks |
| Batch Scoring | Upload a CSV (template provided), score up to 20,000 machines, filter, download; accuracy metrics if a `Machine failure` column is present |
| History | Every saved prediction; search, filter by risk, delete, clear, export CSV |
| Explainability | Global SHAP importance, CART importance and full rule set, GAM shape functions, MARS basis functions, stacking weights |
| Model Lab | All experiment results: comparison with bootstrap CIs, deployed-model metrics, BMA weights, imbalance, calibration, uncertainty/deferral, ablation, leave-one-type-out, figure gallery |
| Data Explorer | Feature statistics, histograms by outcome, failure-region scatter plots, paged raw records |
| Experiments | Retrain the deployed models or re-run all experiments as background jobs with live logs |

The sidebar switch chooses the feature set (engineered = with temperature difference, power and wear × torque; raw = the
six original inputs). Prediction history is stored in `data/app/history.jsonl`; set `MONGODB_URI` in `.env` (see
`.env.example`) to use MongoDB instead. With Docker: `docker compose up --build` starts the app and MongoDB.

Risk levels: High if the calibrated probability is at or above the validation-chosen threshold, Medium from one third of
the threshold, Low below that, and Uncertain when the probability is close to a coin flip (entropy ≥ 0.8 bits) or the
input lies outside the training data. "Engineer review recommended" applies the deferral policy of Experiment 6
(entropy or model disagreement above the 90th percentile of the validation split).

Front-end development (needs Node.js 18+): `cd frontend && npm install && npm run dev` (proxies `/api` to port 8000).
Model files are saved with a SHA-256 manifest and are only loaded if the hash and library versions match; they are
trained locally and never downloaded.


## Planned architecture

```
                 data/raw/ai4i2020.csv
                          │
            ml/data  ── ingestion → validation (schema, leakage guard) → preprocessing / feature engineering
                          │
            ml/models ── CART · AdaBoost · Gradient Boosting · XGBoost · MARS · RBF · GAM
                          │                   Random Forest · Soft Voting · Stacking · Bayesian Model Averaging
            ml/evaluation ── metrics · CV · calibration · uncertainty · ablation
            ml/explainability ── tree rules · GAM shape functions · SHAP
            ml/inference ── prediction service · risk estimator (Low / Medium / High / Uncertain)
                          │
            backend (FastAPI + MongoDB)  ──  REST API, prediction history, model/experiment metadata
                          │
            frontend (React + TypeScript)  ──  dashboard, model lab, comparison, prediction centre, explainability
```

The ML package never imports the web layer, so every model can be trained, tested and evaluated from the command
line or notebooks alone.

## Getting the dataset

1. Download the AI4I 2020 Predictive Maintenance dataset from UCI:
   https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
2. Unzip and place `ai4i2020.csv` at `data/raw/ai4i2020.csv` (this folder is not tracked by Git).
3. Check it: `python -m ml.data.ingestion` should report 10,000 rows, 14 columns and a failure rate of 0.0339.
   The reference MD5 is `f0adaa9c5e25366c6df1cc2684479871`; a warning is logged if your copy differs.

## Setup and running (macOS / Linux)

Quick way: from this folder run `bash run_mac.sh`. It creates `.venv`, installs the requirements, checks the dataset,
runs the tests and then all experiments.

Step by step:

```bash
cd ~/Documents/ML_Course_Projects/MaintenAI
python3 --version                    # needs 3.10 – 3.13
brew install libomp                  # macOS only: XGBoost needs the OpenMP runtime
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ml.data.ingestion          # 10,000 rows, failure rate 0.0339
python -m pytest                     # 28 tests (ML + API)
python -m ml.experiments.run_all     # all experiments, ~5 min; writes artifacts/metrics/results.json and reports/figures/
python notebooks/make_01_eda.py      # optional: rebuilds the EDA notebook
```

Open the results with `open reports/figures` (macOS) or view `artifacts/metrics/results.json`.

## Project documents

| File | Purpose |
|---|---|
| `PROJECT_STATUS.md` | Current milestone, completed and pending work, decisions, next steps |
| `RESEARCH_NOTES.md` | Research questions, literature, data dictionary, leakage rules, methodology |
| `EXPERIMENT_LOG.md` | Every experiment with configuration and measured results |

## Important limitations

AI4I 2020 is a **synthetic** dataset whose failure labels follow documented threshold rules. Results on it show how
well methods recover those rules; they do not prove performance on real machines. MaintenAI is a decision-support
prototype and does not guarantee failure prevention.
