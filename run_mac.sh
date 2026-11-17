#!/usr/bin/env bash
# Set up and run MaintenAI end to end (macOS / Linux).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), f"Python 3.10-3.13 needed, found {sys.version.split()[0]}"'
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null && ! brew list libomp >/dev/null 2>&1; then
  echo "Installing libomp (needed by XGBoost)..."; brew install libomp
fi
[[ -d .venv ]] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
[[ -f data/raw/ai4i2020.csv ]] || { echo "Missing data/raw/ai4i2020.csv - see README 'Getting the dataset'"; exit 1; }
python -m ml.data.ingestion
python -m pytest -q
python -m ml.experiments.run_all
echo "Done. Results: artifacts/metrics/results.json  Figures: reports/figures/"
